"""End-to-end Qwen3.6-27B prefill on vLLM model-runner-v2 + TAPID persistent kernel.

Prefill-only door (see gpu_daemon/docs/vllm_integration.md): TAPID holds the
only full copy of the decoder weights and runs the 64-layer prefill program;
vLLM keeps the skeleton (embed / final norm / lm_head) and samples. Decode is
out of scope — measure prefill with --max-tokens 1.
"""

from __future__ import annotations

import argparse
import os
import sys

MODEL = None  # no default: pass --model /path/to/Qwen3.6-27B


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Local HF checkpoint dir of Qwen3.6-27B (must "
                        "contain model.safetensors.index.json)")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Repeatable. One request per generate call: the TAPID path runs "
        "a single fresh prefill per step.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Tokens to sample per prompt. The TAPID path refuses decode "
        "steps, so anything above 1 fails after the prefill.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Defaults to max-model-len: a prompt must prefill in one step "
        "(chunked prefill is refused), and must stay within the kernel's "
        "10240-row buffer.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="vLLM derives the KV-cache budget from this "
                        "fraction minus everything the process holds — and "
                        "TAPID's arena (~45GiB) + pools (~16GiB) are part of "
                        "that. Default: auto-computed from the device's free "
                        "memory at startup minus a 1%% margin, so small "
                        "co-resident processes don't trip vLLM's free-memory "
                        "check (0.99 was rejected at 78.32/79.25 GiB free).")
    parser.add_argument("--no-tapid", action="store_true",
                        help="Run the plain vLLM model (baseline).")
    parser.add_argument(
        "--verify", action="store_true",
        help="Ignored: the single-weight-copy design leaves no vLLM decoder "
        "to compare against. Numerics are validated by the TAPID st suite.",
    )
    parser.add_argument(
        "--probe", default="full",
        help="Ignored: single-layer probes were part of the two-copy design.",
    )
    parser.add_argument("--probe-layer", type=int, default=0)
    parser.add_argument("--dump-tokens", default="", help="write token ids to JSON")
    parser.add_argument(
        "--state-audit", action="store_true",
        help="Ignored: TAPID writes no vLLM KV/GDN state in the prefill door.",
    )
    args = parser.parse_args()

    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = args.max_model_len
    if args.verify or args.state_audit or args.probe != "full":
        print(
            "NOTE: --verify/--probe/--state-audit are ignored by the "
            "prefill-only TAPID door.",
            file=sys.stderr,
        )
    if args.max_tokens > 1:
        print(
            "NOTE: --max-tokens > 1 will fail on the first decode step; the "
            "TAPID door measures prefill only.",
            file=sys.stderr,
        )

    from vllm import LLM, SamplingParams

    if args.gpu_memory_utilization is None:
        # vLLM refuses to start when free memory < util x total. Size the
        # budget from what is actually free right now (co-resident processes
        # included) minus a 1% margin; TAPID's arena and pools are allocated
        # later, inside load_model, so they never eat into this check.
        import torch

        free, total = torch.cuda.mem_get_info()
        args.gpu_memory_utilization = max(0.50, free / total - 0.01)
        print(
            f"gpu-memory-utilization auto: {args.gpu_memory_utilization:.4f} "
            f"(free {free / (1 << 30):.2f} / {total / (1 << 30):.2f} GiB)",
            file=sys.stderr,
        )

    additional_config = (
        {}
        if args.no_tapid
        else {
            "tapid": {
                "model_signature": "qwen3_5_dense_27b_bf16",
            }
        }
    )

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # A cached-prefix prefill arrives with tokens already computed, which
        # the TAPID path refuses; every prompt must prefill fresh.
        enable_prefix_caching=False,
        # The checkpoint resolves to the multimodal wrapper. Without this,
        # memory profiling runs the vision tower on dummy images; on the
        # 80GB A100 that collides with TAPID's arena+pools. Limits of 0 make
        # the encoder budget empty, so encoder profiling is skipped.
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config=additional_config,
    )
    out = llm.generate(
        args.prompt or ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    for o in out:
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)
    if args.dump_tokens:
        import json
        json.dump(
            [
                {"prompt": o.prompt,
                 "token_ids": list(o.outputs[0].token_ids),
                 "text": o.outputs[0].text}
                for o in out
            ],
            open(args.dump_tokens, "w"),
        )
        print(f"tokens -> {args.dump_tokens}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    sys.exit(main())
