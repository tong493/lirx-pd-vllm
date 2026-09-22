"""Plain-vLLM prefill baseline (no TAPID): bf16 weights, TF32 disabled.

The comparison point for the TAPID door (run_tapid_vllm.py). TF32 is killed
process-wide via NVIDIA_TF32_OVERRIDE=0 (inherited by every TP worker), and
the parent-side torch.backends toggles are belt and suspenders. The dtype is
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
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1,
                        help="bf16 27B (~54 GiB) fits one 80 GB card; keep "
                        "the baseline on the TAPID door's one-card footprint.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16",
                        help="bfloat16 (default) matches the TAPID door's "
                        "weight dtype. float32 is refused: the model's GDN "
                        "layers reject it "
                        "('ChunkGatedDeltaRuleFunction does not support "
                        "float32'). TF32 stays disabled either way, so "
                        "bfloat16 matmuls run at true bf16 on the A100.")
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
    )

    prompts = args.prompt or ["The capital of France is"]
    for prompt in prompts:
        started = time.monotonic()
        out = llm.generate(
            [prompt],
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        )
        seconds = time.monotonic() - started
        tokens = len(out[0].prompt_token_ids)
        print(
            f"BASELINE prefill: tokens={tokens} wall={seconds:.4f}s "
            f"({tokens / seconds:.0f} tok/s)"
        )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    sys.exit(main())
