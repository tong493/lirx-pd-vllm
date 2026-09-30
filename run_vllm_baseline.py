"""Plain-vLLM baseline (no TAPID): bf16 weights.

--max-tokens accepts a comma list (e.g. 15,20,30): 1 times prefill only;
larger values time the full generate (prefill + decode), with the
ttft/decode split printed on a TIMING line identical to run_tapid_vllm.py's.

The comparison point for the TAPID door (run_tapid_vllm.py). NVIDIA_TF32_OVERRIDE=0
and the torch.backends toggles are inert for a bf16 run — TF32 only ever
applies to fp32 matmuls, and a bf16 model has essentially none — kept as
harmless insurance from the abandoned FP32 plan. The dtype is
bfloat16 — matching the TAPID door's weight config; float32 is refused
because the model's GDN layers reject it
(ChunkGatedDeltaRuleFunction does not support float32).

Memory math: 27B at bf16 is ~54 GiB of weights, which fits a single
80 GB A100 with room for activations/KV — default --tp 1 keeps the baseline
on the same one-card footprint as the TAPID door.

wall= includes a few ms of engine scheduling/sample overhead on top of the
prefill itself; the TAPID door= number is the submit->fetch time only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")  # before any CUDA init


def print_timing(out) -> None:
    """Same TIMING line format as run_tapid_vllm.py, so baseline vs TAPID
    numbers are directly comparable."""
    m = out.metrics
    n = len(out.outputs[0].token_ids)
    if m is None:
        print("TIMING: (unavailable — log_stats disabled)")
        return
    queue = m.scheduled_ts - m.queued_ts
    prefill = m.first_token_ts - m.scheduled_ts
    decode = m.last_token_ts - m.first_token_ts
    infer = m.last_token_ts - m.scheduled_ts
    rate = (n - 1) / decode if n > 1 and decode > 0 else float("nan")
    print(f"TIMING: tokens={n} queue={queue * 1e3:.1f}ms "
          f"ttft={prefill * 1e3:.1f}ms decode={decode * 1e3:.1f}ms "
          f"infer={infer * 1e3:.1f}ms ({rate:.1f} tok/s)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Local HF checkpoint dir of Qwen3.6-27B")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Repeatable. Each prompt is timed as its own single-request "
        "generate call, matching the TAPID door's one-prefill-per-step.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-tokens",
        default="1",
        help="Comma-separated generation budgets (e.g. 15,20,30): each "
        "value is one generate per prompt / bench length, so a single "
        "launch sweeps several generation lengths. 1 (default) = prefill "
        "only.",
    )
    parser.add_argument(
        "--bench-tokens",
        default=None,
        help="Comma-separated exact token lengths (e.g. 4,64,1024). Each "
        "length is timed --bench-reps times as a single-request generate "
        "with a TokensPrompt of that exact length and the same filler ids "
        "run_tapid_vllm.py uses, so the two benches see identical inputs.",
    )
    parser.add_argument(
        "--bench-reps", type=int, default=3,
        help="Repetitions per length in --bench-tokens mode.",
    )
    parser.add_argument("--tp", type=int, default=1,
                        help="bf16 27B (~54 GiB) fits one 80 GB card; keep "
                        "the baseline on the TAPID door's one-card footprint.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16",
                        help="bfloat16 (default) matches the TAPID door's "
                        "weight dtype. float32 is refused: the model's GDN "
                        "layers reject it "
                        "('ChunkGatedDeltaRuleFunction does not support "
                        "float32').")
    args = parser.parse_args()

    if args.dtype == "float32":
        parser.error(
            "--dtype float32 is not supported: the GDN layers "
            "(ChunkGatedDeltaRuleFunction) reject FP32. Use bfloat16 — that "
            "is also what the TAPID door runs."
        )

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Same constraint as the TAPID path: prefix caching would let a
        # repeated prompt skip recompute and fake the prefill time.
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        # The offline LLM entrypoint defaults disable_log_stats=True
        # (vllm/entrypoints/llm.py), which leaves RequestOutput.metrics as
        # None; the TIMING lines below need RequestStateStats.
        disable_log_stats=False,
    )

    if args.bench_tokens:
        try:
            from vllm.inputs import TokensPrompt
        except ImportError:
            from vllm import TokensPrompt
        max_tokens_list = [int(x) for x in str(args.max_tokens).split(",")
                           if x.strip()]
        # Same filler ids and timing shape as run_tapid_vllm.py's serial
        # bench: one exact-length TokensPrompt per generate call, so the two
        # measurements differ only in the engine behind the door.
        for raw in args.bench_tokens.split(","):
            n = int(raw.strip())
            for mt in max_tokens_list:
                sampling = SamplingParams(temperature=0.0, max_tokens=mt)
                for rep in range(args.bench_reps):
                    prompt = TokensPrompt(
                        prompt_token_ids=[2000 + (n % 100)] * n)
                    t0 = time.perf_counter()
                    # NVTX range for nsys/plot_prefill_load.py (host-side
                    # marker only, no GPU-side effect). "prefill" is only the
                    # honest name when max_tokens==1; longer generations are
                    # prefill+decode.
                    name = "prefill" if mt <= 1 else "generate"
                    torch.cuda.nvtx.range_push(name)
                    out = llm.generate([prompt], sampling)
                    torch.cuda.nvtx.range_pop()
                    wall = time.perf_counter() - t0
                    print(
                        f"BASELINE bench length={n} max_tokens={mt} "
                        f"rep={rep}: wall={wall:.3f}s "
                        f"({n / wall:.0f} tok/s incl. overhead)"
                    )
                    print_timing(out[0])
        return 0

    max_tokens_list = [int(x) for x in str(args.max_tokens).split(",")
                       if x.strip()]
    prompts = args.prompt or ["The capital of France is"]
    for prompt in prompts:
        for mt in max_tokens_list:
            started = time.monotonic()
            out = llm.generate(
                [prompt],
                SamplingParams(temperature=0.0, max_tokens=mt),
            )
            seconds = time.monotonic() - started
            tokens = len(out[0].prompt_token_ids)
            gen = len(out[0].outputs[0].token_ids)
            print(
                f"BASELINE generate: prompt_tokens={tokens} "
                f"max_tokens={mt} (generated {gen}) wall={seconds:.4f}s "
                f"({gen / seconds:.1f} gen tok/s incl. overhead)"
            )
            print_timing(out[0])
            print(f"  text: {out[0].outputs[0].text!r}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    sys.exit(main())
