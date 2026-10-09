# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TAPID prefill runner: vLLM drives TAPID's existing host-side door.

Contract (see gpu_daemon/docs/vllm_integration.md in the TAPID repo):

* Single weight copy. ``load_model`` dummy-loads the model, restores real
  weights for the skeleton only (token embedding, final norm, lm_head), frees
  every other parameter, and uploads the full decoder into TAPID's arena via
  ``bind_weights_from_host``. TAPID's arena is the only full copy in memory.
* Only TAPID's host-side instructions are called (``tapid_vllm`` ->
  pyshim C ABI): open, bind, launch, set_program, submit (F32), fetch. No
  runtime/KV-cache binding: the submit/fetch path owns its device buffers.
* Prefill only, unless decode mode is on (``additional_config["tapid"]["decode"]``).
  Decode mode arms TAPID's device-side decode loop (issue #190): prefill runs
  the output_head program (terminal output [1, 2] = sampled token id), the
  generated token feeds back to the entry stage ON DEVICE, and the daemon
  keeps producing tokens until eos or budget. vLLM's sampler is bypassed —
  the token the host reports is the one the device sampled. Decode sequences
  submit under request_id == slot; the slot pool and per-slot KV regions are
  TAPID's own (kv_alloc/kv_release), allocated BEFORE the prefill is
  submitted. Decode steps are never submitted: the runner only drains
  produced tokens via fetch_with_token_ids.

Model specifics (checkpoint conversion, program assembly, skeleton names) live
in the TAPID repo's model assembly package; nothing here knows the model.
"""

import gc
import importlib
import os
import time
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

_TAPID_DOOR_MODULE = "tapid_vllm"
# Model-side half of the contract, on PYTHONPATH as $TAPID_REPO.
_TAPID_ADAPTER_MODULE = "models.model_assembly.qwen3_6_27b_dense.vllm_adapter"

# Weight upload of ~53 GiB and a set_program over the resident kernel can take
# minutes on a loaded host; nothing here should time out before they finish.
_TAPID_BIND_TIMEOUT_MS = 1_800_000
_TAPID_PROGRAM_TIMEOUT_MS = 600_000
_TAPID_FETCH_TIMEOUT_MS = 600_000
# The long fetch timeout used to sit silent for its full duration. Chunk it:
# every chunk that returns nothing logs a heartbeat naming the request still
# pending, and the total deadline ends in a device-side scheduling snapshot
# (tapid_debug_dump, twice — what did not move is the stall) before raising.
_TAPID_FETCH_CHUNK_MS = 30_000
_TAPID_DUMP_GAP_S = 5.0
# Decode-mode mailbox waits: a healthy daemon answers a decode step in
# milliseconds, so the chunk is short and the heartbeat frequent; the total
# deadline only exists to convert a wedged device loop into a loud error.
_TAPID_DECODE_CHUNK_MS = 5_000
_TAPID_DECODE_TIMEOUT_MS = 120_000

# Pre-arm the fetch can block far longer than a decode step ever would; a
# hung prefill surfaces as this timeout, not as a silent wedge.
_TAPID_PREFILL_LOG_EVERY = 1
# Merged fresh prefills per step: vLLM's continuous batching admits several
# requests at once; the door submits each request as its own TAPID batch
# (own input buffer / terminal slot). The per-request computed==0 and row
# checks in _prefill_batch are what actually guard the door; this only
# bounds how wide one step's fan-out may be.
_TAPID_MAX_REQS_PER_STEP = 8


def validate_tapid_config(runner: Any) -> None:
    tapid_config = runner.vllm_config.additional_config["tapid"]
    if tapid_config.get("model_signature") != "qwen3_5_dense_27b_bf16":
        raise ValueError("TAPID requires the Qwen3.5 27B BF16 signature")
    runner.tapid_decode = bool(tapid_config.get("decode", False))
    if runner.tapid_decode:
        if os.environ.get("TAPID_SKIP_LM_HEAD") == "1":
            raise ValueError(
                "TAPID decode mode samples on device (head SOP argmax); "
                "TAPID_SKIP_LM_HEAD=1 is meaningless and forbidden here"
            )
        if runner.model_config.hf_text_config.tie_word_embeddings:
            raise ValueError(
                "TAPID decode mode binds lm_head INSIDE TAPID "
                "(load_decode_tail_weights), which needs untied embeddings"
            )
    if not runner.model_config.enforce_eager:
        raise ValueError("TAPID requires enforce_eager")
    if runner.model_config.hf_text_config.model_type != "qwen3_5_text":
        raise ValueError("TAPID requires the dense Qwen3.5 text model")
    if runner.model_config.dtype != torch.bfloat16:
        raise ValueError("TAPID requires bfloat16")
    if (
        runner.parallel_config.tensor_parallel_size != 1
        or runner.parallel_config.pipeline_parallel_size != 1
    ):
        raise ValueError("TAPID P0/P1 requires TP=1 and PP=1")
    if runner.speculative_config is not None or runner.lora_config is not None:
        raise ValueError("TAPID P0/P1 does not support spec decode or LoRA")
    if runner.parallel_config.enable_dbo:
        raise ValueError("TAPID P0/P1 does not support DBO")
    if runner.vllm_config.quant_config is not None:
        raise ValueError("TAPID P0/P1 does not support quantization")

    adapter = runner.tapid_adapter
    # The door splits a step per request — each request becomes its own TAPID
    # batch, checked against MAX_PREFILL_TOKENS at runtime — so the step
    # budget and max_model_len may both exceed the row buffer (4x1024 needs
    # 4096). What must hold statically: the budget admits a whole sequence,
    # else vLLM chunk-prefills, which the door refuses.
    max_batched = int(runner.scheduler_config.max_num_batched_tokens)
    if runner.model_config.max_model_len > max_batched:
        raise ValueError(
            f"max_num_batched_tokens={max_batched} < max_model_len="
            f"{runner.model_config.max_model_len} would chunk prefill, which "
            "TAPID refuses. Raise --max-num-batched-tokens (it may exceed "
            "the kernel's 2560-row buffer: each request is submitted as its "
            "own TAPID batch, bounded per request at runtime)."
        )


_TORCH_SYNC_ORIGINALS: dict[str, Any] = {}


def _install_stream_only_sync(device: torch.device) -> None:
    """Downgrade device-wide syncs to a sync of the current stream.

    TAPID's persistent kernels never return, so anything that waits for *all*
    work on the device — ``cudaDeviceSynchronize`` under
    ``torch.cuda.synchronize`` / ``torch.accelerator.synchronize`` — blocks
    forever once they are resident. vLLM only ever needs its own submitted work
    to have landed, which a stream sync gives it.

    ponytail: syncs the current stream only. vLLM's side streams (async output
    copy, prefetch) are already joined through events; add an explicit
    per-stream list here if a future call site needs one that is not.
    """
    if _TORCH_SYNC_ORIGINALS:
        return

    def _sync_current_stream(*args: Any, **kwargs: Any) -> None:
        torch.cuda.current_stream(device).synchronize()

    for module in (torch.cuda, torch.accelerator):
        _TORCH_SYNC_ORIGINALS[module.__name__] = module.synchronize
        module.synchronize = _sync_current_stream
    logger.info("TAPID: device-wide CUDA syncs downgraded to current-stream syncs")


def _restore_device_sync() -> None:
    for module in (torch.cuda, torch.accelerator):
        original = _TORCH_SYNC_ORIGINALS.pop(module.__name__, None)
        if original is not None:
            module.synchronize = original


def _install_spin_wait_output_event() -> None:
    """Swap AsyncOutput's blocking copy event for a spin-wait one.

    The engine waits on ``torch.cuda.Event(blocking=True)`` (the
    cudaEventBlockingSync flavor: the host thread sleeps on a driver
    interrupt) for the async D2H output copy. A spin-wait event polls the
    GPU-side completion instead, avoiding any host-side sleep primitive that
    might interact badly with the resident persistent kernel.
    """
    import vllm.v1.worker.gpu.async_utils as async_utils_mod

    original_init = async_utils_mod.AsyncOutput.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_event = torch.cuda.Event

        class _SpinEvent(original_event):  # type: ignore[name-defined]
            def __init__(self, *event_args: Any, **event_kwargs: Any) -> None:
                event_kwargs.pop("blocking", None)
                super().__init__(*event_args, **event_kwargs)

        torch.cuda.Event = _SpinEvent
        try:
            original_init(self, *args, **kwargs)
        finally:
            torch.cuda.Event = original_event

    async_utils_mod.AsyncOutput.__init__ = patched_init  # type: ignore[assignment]
    logger.info("TAPID: AsyncOutput copy event swapped to spin-wait")


def _import_tapid_modules() -> tuple[Any, Any]:
    try:
        door = importlib.import_module(_TAPID_DOOR_MODULE)
        adapter = importlib.import_module(_TAPID_ADAPTER_MODULE)
    except ImportError as exc:
        raise ImportError(
            "TAPID integration requires $TAPID_REPO and $TAPID_REPO/python on "
            "PYTHONPATH (run_e2e.sh sets both; see "
            "gpu_daemon/docs/vllm_integration.md)"
        ) from exc
    return door, adapter


class TapidGPUModelRunner(GPUModelRunner):
    """V1 runner is not part of the prefill-only door.

    gpu_worker selects this class when the V2 runner is off; constructing it
    fails loudly instead of silently running the old two-copy integration.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        raise NotImplementedError(
            "The TAPID prefill door only supports the V2 model runner; set "
            "VLLM_USE_V2_MODEL_RUNNER=1 (run_e2e.sh does)."
        )


class TapidGPUModelRunnerV2(GPUModelRunnerV2):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.tapid_door, self.tapid_adapter = _import_tapid_modules()
        validate_tapid_config(self)

        self.tapid_session: Any = None
        self.tapid_armed = False
        # Keepalive: the host tensors the weight upload copied into TAPID's
        # arena. Freed only when the session closes.
        self._tapid_decoder_weights: Any = None
        self._tapid_hidden_size = self.model_config.get_hidden_size()
        self._tapid_next_request = 0
        self._tapid_steps = 0
        # Decode mode (issue #190). Slots are TAPID-side: request_id == slot
        # for decode sequences, kv_alloc before the prefill submit, kv_release
        # when vLLM finishes the request. Drained tokens buffer per slot
        # because the daemon produces on device while the engine steps at its
        # own cadence — one token is reported per engine step.
        self._tapid_slot_free: list[int] = []
        self._tapid_req_slot: dict[Any, int] = {}
        # Paged KV: last page count mirrored per slot (rows are append-only
        # within a generation, so an unchanged count means an unchanged row).
        self._tapid_slot_page_count: dict[int, int] = {}
        self._tapid_pending: dict[int, list[int]] = {}
        self._tapid_awaiting_first: set[int] = set()
        # Slots whose device loop reached EOS/budget: the daemon stopped
        # feeding, so no further mailbox entries will ever arrive for them.
        self._tapid_done_slots: set[int] = set()
        self._tapid_generated: dict[int, int] = {}
        self._tapid_step_slots: list[int | None] = []
        # The real-step InputBatch, stashed by the prepare_attn hook below:
        # the only reliable per-step request state (req_ids and the
        # scheduler-fed num_computed_tokens_np in batch order). The forward
        # context's attention metadata is backend-specific (FlashAttention
        # metadata here) and its seq_lens is GPU-side, fed by
        # req_states.num_computed_tokens.gpu — which decode mode leaves stale.
        self._tapid_input_batch: Any = None
        self._tapid_step_req_ids: list[Any] = []
        self._tapid_eos_id = 0
        self._tapid_max_len = 0

    # ---- model load: the single weight copy -------------------------------

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        # Always dummy-load, regardless of the caller's flag: the vLLM model
        # keeps only the skeleton, so a full real load would transiently hold
        # a second copy of the decoder that we would free again right after.
        if not load_dummy_weights:
            logger.info_once(
                "TAPID: forcing dummy weight load; the decoder weights come "
                "from TAPID's own upload and only the skeleton stays in vLLM"
            )
        super().load_model(True, *args, **kwargs)
        self._load_tapid_skeleton()
        self._free_decoder_weights()
        self._bind_tapid_weights()

    def _load_tapid_skeleton(self) -> None:
        """Restore real values for embed / final norm / lm_head.

        TAPID's submit takes token embeddings as its input (so vLLM embeds,
        TAPID decodes); the program boundary is the decoder output (so vLLM
        applies the final norm); and vLLM samples from lm_head. Names are the
        checkpoint's own; the model's ``hf_to_vllm_mapper`` strips the VL-era
        ``model.language_model.`` prefix.
        """
        skeleton = self.tapid_adapter.load_skeleton_tensors(
            self.model_config.model
        )
        tied = bool(
            getattr(self.model_config.hf_text_config, "tie_word_embeddings", False)
        )
        weights = {
            name: tensor
            for name, tensor in skeleton.items()
            if not (tied and "lm_head" in name)
        }
        if not tied and not any("lm_head" in name for name in weights):
            raise ValueError(
                "lm_head is untied but the checkpoint skeleton has no "
                "lm_head tensor; cannot sample without it"
            )
        loaded = self.model.load_weights(iter(weights.items()))
        logger.info(
            "TAPID: skeleton weights restored (%s), tied_lm_head=%s",
            sorted(loaded or []), tied,
        )

    def _free_decoder_weights(self) -> None:
        """Release the decoder-layer parameters back to the allocator.

        The persistent-kernel program is the only consumer of the decoder
        weights, so the vLLM copies are dead weight (~50 GiB). Decoder layer
        params (``.layers.``) and the vision tower (``.visual.``) are freed;
        the embedding, final norm, and lm_head stay real.

        Params become empty meta tensors: any accidental access fails loudly
        instead of silently computing on dummy values. Some params are tensor
        subclasses (e.g. DTensor) whose ``set_data`` refuses a plain meta
        tensor; for those the registered attribute is replaced wholesale.
        """
        freed_bytes = 0
        skipped: list[str] = []
        with torch.no_grad():
            for name, param in list(self.model.named_parameters()):
                if name.endswith(
                    ("embed_tokens.weight", "lm_head.weight", "model.norm.weight")
                ):
                    continue
                if ".layers." not in name and ".visual." not in name:
                    continue
                meta_empty = torch.empty(
                    0, dtype=param.dtype, device="meta"
                )
                try:
                    param.data = meta_empty
                except RuntimeError:
                    # Subclass params: swap the registered attribute itself.
                    try:
                        parent_name, leaf = name.rsplit(".", 1)
                        parent = self.model.get_submodule(parent_name)
                        setattr(
                            parent,
                            leaf,
                            torch.nn.Parameter(
                                meta_empty, requires_grad=False
                            ),
                        )
                    except Exception as exc:
                        skipped.append(f"{name} ({type(param).__name__}: {exc})")
                        continue
                freed_bytes += param.numel() * param.element_size()
        logger.info(
            "TAPID: freed %.1f GiB of decoder parameters from the vLLM model",
            freed_bytes / (1 << 30),
        )
        if skipped:
            logger.info(
                "TAPID: %d params kept real (unfreable): %s",
                len(skipped), "; ".join(skipped[:8]),
            )
        # The freed tensors only return to torch's caching allocator; until
        # they are released back to the driver, TAPID's cudaMemGetInfo-based
        # arena sizing sees no free memory and refuses the weight upload.
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        logger.info(
            "TAPID: device free after release: %.1f / %.1f GiB "
            "(the arena upload needs the decoder weights' worth)",
            free_bytes / (1 << 30), total_bytes / (1 << 30),
        )

    def _bind_tapid_weights(self) -> None:
        """Convert the checkpoint once and upload it into TAPID's arena.

        This happens inside load_model on purpose: the arena's device memory
        is allocated before vLLM profiles free memory, so KV-cache sizing
        automatically accounts for it. The adapter's layout conversions are
        the same ones the TAPID st suite validates against reference outputs.
        """
        started = time.monotonic()
        weights = self.tapid_adapter.load_decoder_weights(self.model_config.model)
        if self.tapid_decode:
            # Final norm + lm_head + embed table must live INSIDE TAPID: the
            # output_head program executes them on device and the decode loop
            # gathers fed-token rows from the embed binding itself.
            tail = self.tapid_adapter.load_decode_tail_weights(
                self.model_config.model
            )
            logger.info("TAPID decode: binding %d decode-tail weights", len(tail))
            weights = weights + tail
        # Keepalive for the host tensors the upload borrows.
        self._tapid_decoder_weights = weights
        self.tapid_session = self.tapid_door.TapidVllmSession([self.device.index])
        self.tapid_session.bind_weights_from_host(
            0, weights, timeout_ms=_TAPID_BIND_TIMEOUT_MS
        )
        logger.info(
            "TAPID: %d decoder weights converted and uploaded (%.1f GiB, "
            "%.1fs including conversion)",
            len(weights),
            self.tapid_adapter.arena_nbytes(weights) / (1 << 30),
            time.monotonic() - started,
        )
        # Last kernel launches before the persistent kernel goes resident --
        # a CUDA module loaded lazily *after* that never finishes loading.
        self._warm_tapid_kernels()

    # ---- kernel warmup -----------------------------------------------------

    def _warm_tapid_kernels(self) -> None:
        """Force every CUDA module the armed path may launch to load NOW.

        ``CUDA_MODULE_LOADING`` defaults to LAZY, so a module loads on its
        first launch. TAPID's persistent kernels never exit, so a first launch
        after they go resident blocks inside the driver forever. vLLM's own
        warmup does not cover this runner's armed path, because the pre-arm
        stub skips the pieces that only the armed path needs at every token
        count.

        The token count picks the embedding's kernel specialisation, so the
        sweep must cover the lengths the engine can submit (measured, not
        guessed: a 6-token prompt after a 35-token one once hung inside
        F.embedding's index_select). Everything else (casts, copies, the final
        norm) dispatches independently of token count, but sweeping costs
        nothing extra here.
        """
        embed = getattr(self.model, "embed_input_ids", None)
        if embed is None:
            raise RuntimeError(
                "TAPID requires the model to expose embed_input_ids"
            )
        max_tokens = int(self.scheduler_config.max_num_batched_tokens)
        counts = {1, max_tokens}
        n = 2
        while n < max_tokens:
            counts.add(n)
            n *= 2
        text_norm = self._tapid_text_model().norm
        for count in sorted(counts):
            ids = torch.zeros(count, dtype=torch.int32, device=self.device)
            hidden = embed(ids)
            # Exactly the staging ops the armed path performs, dispatch-for-
            # dispatch: bf16->F32 cast folded into the D2H copy, then H2D copy
            # with the cast back to BF16, then the final norm.
            staged = hidden.to("cpu", dtype=torch.float32)
            dev = staged.to(self.device, dtype=self.model_config.dtype)
            text_norm(dev)
        # Greedy sampling: the engine's sampler warmup uses temperature=0.9,
        # which marks the batch all_random and skips greedy_sample entirely —
        # so torch.argmax stays lazily unloaded until the first temperature=0
        # request lands on it post-arm, blocking on the resident kernel.
        vocab_size = self.model_config.get_vocab_size()
        dummy_logits = torch.zeros(
            (2, vocab_size), dtype=torch.float32, device=self.device
        )
        torch.argmax(dummy_logits, dim=-1)
        # The logits GEMM: cuBLAS picks a different kernel per (M, N, K) and
        # loads that kernel's module inside the library, which
        # CUDA_MODULE_LOADING=EAGER does not cover. The real request samples
        # M=1 rows (one per request); sweep a few M values so no post-arm
        # launch lands on a lazily-loaded cuBLAS kernel.
        hidden = self._tapid_hidden_size
        for rows in (1, 2, 4, 8):
            dummy_hidden = torch.zeros(
                (rows, hidden), dtype=self.model_config.dtype, device=self.device
            )
            self.model.compute_logits(dummy_hidden)
            torch.cuda.synchronize()
        logger.info(
            "TAPID: preloaded embed/stage/norm kernels for %d token counts, "
            "the greedy argmax kernel, and the logits GEMM for M in (1, 2, 4, 8)",
            len(counts),
        )

    # ---- arming ------------------------------------------------------------

    def tapid_arm(self) -> None:
        """Hand steady-state prefill over to TAPID (called after vLLM warmup).

        Warmup is full of device-wide syncs, so the persistent kernels only go
        resident here. After this point every sync in the process must be a
        stream sync — installed before the launch, not after.
        """
        if self.tapid_session is None:
            return
        _install_stream_only_sync(self.device)
        _install_spin_wait_output_event()
        started = time.monotonic()
        self.tapid_session.launch()
        # Same settle window the bench runner uses: the resident CTAs come up
        # before the program is swapped in.
        time.sleep(0.3)
        logger.info("TAPID: launch returned in %.2fs", time.monotonic() - started)
        # Review 1.3: the decode program's ENTRY stage is the model's Embed
        # SOP — prefill submissions become [T, 1] token ids (no host-side
        # gather, no [T, hidden] upload). The prefill-measurement program
        # keeps the host-side embedding: its inputs_embeds path has no token
        # ids to feed.
        payload = self.tapid_adapter.decoder_program_payload(
            self.tapid_session.abi, output_head=self.tapid_decode,
            embed_entry=self.tapid_decode,
        )
        self.tapid_session.set_program(
            payload, timeout_ms=_TAPID_PROGRAM_TIMEOUT_MS
        )
        if self.tapid_decode:
            self._configure_tapid_decode()
        self.tapid_armed = True
        if os.environ.get("TAPID_SKIP_LM_HEAD") == "1":
            self._install_fake_logits()
        logger.info(
            "TAPID armed: persistent prefill program resident (%.1fs)",
            time.monotonic() - started,
        )

    def _configure_tapid_decode(self) -> None:
        """Arm kv_cache_mgr + the decode-loop daemon (issue #190).

        max_slots is the terminal pool size (MAX_DECODE_SLOTS == N_TERMINAL in
        the compiled binary) — the decode daemon reads the head SOP's [1, 2]
        mailbox entries through that pool, so a decode sequence must submit
        under request_id == slot. The budget handed to kv_alloc is the region
        limit (max_model_len - prompt_len); vLLM stops the request at its own
        max_tokens earlier, and the extra budget only means the daemon would
        have kept going — drained tokens for a released slot are discarded.
        """
        from tapid.bench.abi import N_TERMINAL

        max_slots = int(self.tapid_session.abi[N_TERMINAL])
        if self.scheduler_config.max_num_seqs > max_slots:
            raise ValueError(
                f"max_num_seqs={self.scheduler_config.max_num_seqs} exceeds "
                f"the decode slot pool ({max_slots} = N_TERMINAL); lower "
                "--max-num-seqs"
            )
        self._tapid_eos_id = self._tapid_resolve_eos()
        self._tapid_max_len = self.model_config.max_model_len
        self.tapid_adapter.configure_decode(
            self.tapid_session,
            max_slots=max_slots,
            max_len=self._tapid_max_len,
            eos_id=self._tapid_eos_id,
            filler=True,
        )
        self._tapid_slot_free = list(range(max_slots))
        logger.info(
            "TAPID decode: %d slots armed, max_len=%d, eos_id=%d",
            max_slots, self._tapid_max_len, self._tapid_eos_id,
        )

    def _tapid_resolve_eos(self) -> int:
        eos = getattr(self.model_config.hf_config, "eos_token_id", None)
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        if eos is None:
            eos = getattr(
                self.model_config.hf_text_config, "eos_token_id", 0
            )
        return int(eos or 0)

    def _install_fake_logits(self) -> None:
        """BENCH ONLY: replace the lm_head GEMM with a zeros tensor.

        Kept for non-bench sampling paths; the bench path below skips the
        sampler entirely.
        """
        vocab = self.model_config.get_vocab_size()

        def zeros_logits(hidden_states: torch.Tensor, *_args: Any,
                         **_kwargs: Any) -> torch.Tensor:
            return torch.zeros(
                (hidden_states.shape[0], vocab),
                dtype=torch.float32,
                device=hidden_states.device,
            )

        self.model.compute_logits = zeros_logits
        logger.warning(
            "TAPID bench: lm_head GEMM bypassed (TAPID_SKIP_LM_HEAD=1); "
            "sampled tokens are meaningless, prefill door= timings stay valid"
        )

    def _bench_sampling_skipped(self) -> bool:
        return (
            self.tapid_armed
            and os.environ.get("TAPID_SKIP_LM_HEAD") == "1"
        )

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        """BENCH ONLY: drop everything after the term-out door.

        The post-prefill machinery is exactly what never runs under the
        resident persistent kernel: the sampler's kernels, the AsyncOutput
        D2H copy (a second stream + event wait -- where the very first
        py-spy pinned this hang), and the triton postprocess kernels. In
        bench mode none of it is needed: fabricate the finished
        ModelRunnerOutput host-side (token id 0 per request; --max-tokens 1
        finishes each request immediately) so the scheduler can line up the
        next prefill while the persistent kernel stays resident.
        """
        if self.tapid_armed and self.tapid_decode:
            return self._decode_sample_tokens()
        if not self._bench_sampling_skipped():
            return super().sample_tokens(grammar_output)
        state = self.execute_model_state
        self.execute_model_state = None
        input_batch = state.input_batch
        req_ids = list(input_batch.req_ids)
        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={
                req_id: i for i, req_id in enumerate(req_ids)
            },
            sampled_token_ids=[[0] for _ in req_ids],
            prompt_logprobs_dict={},
        )
        output.kv_connector_output = self.kv_connector.post_forward(
            state.finished_req_ids
        )
        return output

    def prepare_attn(self, input_batch: Any) -> Any:
        """Stash the real-step InputBatch before the backend builds metadata.

        The TAPID forward paths need req_ids and num_computed_tokens_np in
        batch order; the forward context only exposes backend metadata
        (FlashAttentionMetadata here), which carries neither.
        """
        self._tapid_input_batch = input_batch
        return super().prepare_attn(input_batch)

    # ---- forwards ----------------------------------------------------------
    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> Any:
        if not self.tapid_armed:
            return self._stub_forward(input_ids, inputs_embeds)
        if self.tapid_decode:
            return self._tapid_decode_forward(input_ids, inputs_embeds)
        return self._tapid_prefill_forward(input_ids, inputs_embeds)

    def _stub_forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> Any:
        """Pre-arm stand-in for the model body: embed -> final norm.

        Runs for every profile/dummy/warmup step, so the kernels it needs load
        before the persistent kernel goes resident. It touches only skeleton
        weights — the decoder parameters are already meta-empty, which is safe
        precisely because this stub never walks the layers.
        """
        if inputs_embeds is not None:
            hidden = inputs_embeds
        elif input_ids is not None:
            hidden = self.model.embed_input_ids(input_ids)
        else:
            raise RuntimeError(
                "TAPID stub forward requires token ids or input embeddings"
            )
        normed = self._tapid_text_model().norm(hidden)
        if isinstance(normed, tuple):
            normed = normed[0]
        return normed.to(self.model_config.dtype)

    def _prefill_batch(self) -> tuple[int, int, list, list[int]]:
        """Validate the scheduled batch; return (rows, num_reqs, bounds, computed).

        Everything comes from host-side reads (``query_start_loc.cpu()`` is
        a plain memcpy) — no kernel launches, which are the dangerous ones
        once the persistent kernel is resident. computed[i] is how many
        tokens of request i were already computed: 0 for a fresh prefill,
        >0 for the decode steps decode mode tolerates (the daemon owns
        those rows).

        computed comes from the stashed InputBatch's num_computed_tokens_np
        (batch order, refreshed from the scheduler by update_requests).
        batch_md.seq_lens must NOT be used: it is written by the
        prepare_pos_seq_lens GPU kernel from req_states.num_computed_tokens
        .gpu, which decode mode leaves stale (the sampler bypass skips
        postprocess_num_computed_tokens), so it reads 0 + query_len for
        decode steps.
        """
        context = get_forward_context()
        metadata_map = context.attn_metadata
        if not isinstance(metadata_map, dict):
            raise RuntimeError(
                "TAPID prefill expects the hybrid attention metadata dict"
            )
        batch_md = None
        for key, value in metadata_map.items():
            if key.endswith(".self_attn.attn"):
                batch_md = value
                break
            if batch_md is None and getattr(value, "query_start_loc", None) is not None:
                batch_md = value
        if batch_md is None:
            raise RuntimeError(
                "TAPID prefill could not find query_start_loc in the "
                "attention metadata"
            )

        if int(getattr(batch_md, "num_spec_decodes", 0) or 0) != 0:
            raise RuntimeError("TAPID prefill does not support speculative decode")
        for value in metadata_map.values():
            if int(getattr(value, "num_spec_decodes", 0) or 0) != 0:
                raise RuntimeError(
                    "TAPID prefill does not support speculative decode"
                )

        query_start_loc = batch_md.query_start_loc.cpu()
        num_reqs = query_start_loc.numel() - 1
        if not 1 <= num_reqs <= _TAPID_MAX_REQS_PER_STEP:
            raise RuntimeError(
                f"TAPID prefill runs 1..{_TAPID_MAX_REQS_PER_STEP} requests "
                f"per step, got {num_reqs}"
            )
        rows = int(query_start_loc[-1].item())
        step_batch = self._tapid_input_batch
        if step_batch is None:
            raise RuntimeError(
                "TAPID prefill: no stashed InputBatch — prepare_attn did "
                "not run for this step"
            )
        req_ids = list(step_batch.req_ids)
        if len(req_ids) != num_reqs:
            raise RuntimeError(
                f"TAPID prefill: metadata has {num_reqs} requests but the "
                f"InputBatch has {len(req_ids)}"
            )
        computed_np = step_batch.num_computed_tokens_np
        max_rows = self.tapid_adapter.MAX_PREFILL_TOKENS
        bounds = []
        computed_list: list[int] = []
        for i in range(num_reqs):
            start = int(query_start_loc[i])
            end = int(query_start_loc[i + 1])
            computed = int(computed_np[i])
            if computed != 0 and not self.tapid_decode:
                hint = (
                    "decode step"
                    if end - start == 1
                    else "chunked prefill (run with --max-num-batched-tokens >= "
                    "--max-model-len to prefill in one step)"
                )
                raise RuntimeError(
                    f"TAPID prefill cannot continue a sequence (request {i}: "
                    f"{hint}; {computed} tokens already computed). Decode is "
                    f"out of scope: use --max-tokens 1 to measure prefill only."
                )
            if not 0 < end - start <= max_rows:
                raise RuntimeError(
                    f"TAPID prefill request {i} row count {end - start} "
                    f"outside (0, {max_rows}]"
                )
            bounds.append((start, end))
            computed_list.append(computed)
        self._tapid_step_req_ids = req_ids
        return rows, num_reqs, bounds, computed_list

    def _tapid_prefill_forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> Any:
        rows, num_reqs, bounds, _computed = self._prefill_batch()
        if inputs_embeds is not None:
            hidden = inputs_embeds
        elif input_ids is not None:
            hidden = self.model.embed_input_ids(input_ids)
        else:
            raise RuntimeError(
                "TAPID prefill requires token ids or input embeddings"
            )
        if hidden.shape[0] < rows or hidden.shape[1] != self._tapid_hidden_size:
            raise RuntimeError(
                f"embedded hidden {tuple(hidden.shape)} cannot carry {rows} "
                f"rows of width {self._tapid_hidden_size}"
            )

        # One TAPID batch per request: each request lands in its own input
        # buffer / terminal slot (the request_id selects the pool entry), so
        # concurrent requests' pipelines overlap inside the kernel. Every
        # submit is queued before any fetch waits — tapid_submit_v2 only
        # enqueues; the wait happens in fetch.
        door_started = time.monotonic()
        request_ids: list[int] = []
        for start, end in bounds:
            request_id = self._tapid_next_request
            self._tapid_next_request += 1
            request_ids.append(request_id)
            self.tapid_session.submit_hidden(
                request_id, hidden[start:end], self._tapid_hidden_size
            )
        submit_seconds = time.monotonic() - door_started
        # Marker between the submit and fetch phases: silence BEFORE this line
        # means a submit-stage stall (staging D2H / feeder H2D), silence AFTER
        # means the device never delivered a terminal result.
        logger.info(
            "TAPID: submitted reqs=%s rows=%s (submit=%.4fs)",
            request_ids, [end - start for start, end in bounds],
            submit_seconds,
        )

        fetch_started = time.monotonic()
        flats: list[np.ndarray] = []
        arrivals: list[float] = []
        fetch_deadline = fetch_started + _TAPID_FETCH_TIMEOUT_MS / 1000.0
        for (start, end), request_id in zip(bounds, request_ids):
            n_rows = end - start
            while True:
                try:
                    out_flat, out_rows, out_cols = (
                        self.tapid_session.fetch_array(
                            request_id,
                            n_rows * self._tapid_hidden_size,
                            timeout_ms=_TAPID_FETCH_CHUNK_MS,
                        )
                    )
                    break
                except self.tapid_door.TapidError as exc:
                    if "timed out" not in str(exc):
                        raise
                    now = time.monotonic()
                    if now >= fetch_deadline:
                        logger.error(
                            "TAPID: request %d (rows=%d) produced no terminal "
                            "output after %.0fs — dumping device state",
                            request_id, n_rows, now - fetch_started,
                        )
                        self._dump_tapid_stall(f"fetch req={request_id}")
                        raise
                    logger.info(
                        "TAPID: still waiting for request %d (rows=%d, "
                        "%.0fs elapsed)", request_id, n_rows,
                        now - fetch_started,
                    )
            arrivals.append(time.monotonic() - fetch_started)
            if (out_rows, out_cols) != (n_rows, self._tapid_hidden_size):
                raise RuntimeError(
                    f"TAPID returned {out_rows}x{out_cols} for request "
                    f"{request_id:#x}, expected {n_rows}x"
                    f"{self._tapid_hidden_size}"
                )
            # Keep the result on HOST (writable numpy view over the fetch
            # buffer, freshly allocated per call): the multi-request path
            # must reuse the serial path's exact post-door stream recipe.
            # py-spy pinned the parallel-round wedge to the drain sync below,
            # and the only ops there serial never exercised post-arm were
            # torch.cat and the back-to-back per-request H2D copies — so
            # neither runs anymore: concat on host, ONE copy, ONE norm.
            flats.append(np.asarray(out_flat).reshape(out_rows, out_cols))
        door_seconds = time.monotonic() - door_started
        host = flats[0] if len(flats) == 1 else np.concatenate(flats, axis=0)
        out = torch.from_numpy(host).to(
            self.device, dtype=self.model_config.dtype
        )
        torch.cuda.current_stream().synchronize()
        logger.info("TAPID: post-door H2D drained (rows=%d)", out.shape[0])
        normed = self._tapid_text_model().norm(out)
        if isinstance(normed, tuple):
            normed = normed[0]

        self._tapid_steps += 1
        if self._tapid_steps % _TAPID_PREFILL_LOG_EVERY == 1:
            per_batch = " ".join(f"{x:.3f}" for x in arrivals)
            logger.info(
                "TAPID prefill #%d: rows=%d reqs=%d door=%.4fs "
                "(submit=%.4fs batches=[%s]) (%.0f tok/s)",
                self._tapid_steps, rows, num_reqs, door_seconds,
                submit_seconds, per_batch,
                rows / door_seconds if door_seconds > 0 else 0,
            )
        # Diagnostic for the post-prefill sampling hang: a stream-scoped drain
        # proves whether everything enqueued on the main stream up to here
        # actually executed. Stream syncs are legal post-arm (the patched
        # torch.cuda.synchronize makes engine-level syncs stream-scoped too);
        # a device-wide sync is the thing that must never happen.
        torch.cuda.current_stream().synchronize()
        logger.info(
            "TAPID: main stream drained after prefill #%d", self._tapid_steps
        )
        return normed

    # ---- decode mode (issue #190) ------------------------------------------

    def _tapid_decode_forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> Any:
        """Decode-mode forward: submit fresh prefills, ignore decode rows.

        A fresh prefill (first appearance — the request has no slot yet,
        which only its first scheduled step can be) arms a slot
        (kv_alloc BEFORE submit —
        the prefill's qk_norm_rope commits K/V into the slot's regions at the
        sidecar positions) and submit the embedded hidden rows under
        request_id == slot; the program's head stage lands the first sampled
        token in the slot's terminal mailbox. Prefills submit [T, 1] token ids
        (review 1.3: the program's entry Embed SOP gathers the vocabulary rows
        on-device). Decode rows (an already-armed request's 1-row step) are
        NEVER submitted: the device-side decode loop is already producing that
        request's tokens autonomously; this row exists only so vLLM's scheduler
        bookkeeping advances. The return value is a dummy tensor —
        _decode_sample_tokens below replaces the sampler entirely.
        """
        rows, num_reqs, bounds, computed_list = self._prefill_batch()
        if inputs_embeds is not None:
            raise RuntimeError(
                "TAPID decode mode feeds token ids to the program's entry "
                "Embed SOP; inputs_embeds carries no token ids to gather with"
            )
        if input_ids is None:
            raise RuntimeError("TAPID decode forward requires token ids")
        if input_ids.shape[0] < rows:
            raise RuntimeError(
                f"token ids {tuple(input_ids.shape)} cannot carry {rows} rows"
            )
        from tapid.bench.abi import DTYPE_F32

        step_slots: list[int | None] = []
        n_prefills = 0
        req_ids = self._tapid_step_req_ids
        for i, (start, end) in enumerate(bounds):
            if req_ids[i] in self._tapid_req_slot:
                # Runner-local truth outranks computed here: a request this
                # runner already armed is a decode row the device daemon
                # owns, regardless of what the computed bookkeeping says
                # (the GPU-side copy can lag — see _prefill_batch).
                if end - start != 1:
                    raise RuntimeError(
                        f"TAPID decode request {i} scheduled {end - start} "
                        f"rows for an armed request — chunked prefill is "
                        "unsupported"
                    )
                step_slots.append(None)     # decode row: daemon owns it
                continue
            if computed_list[i] != 0:
                raise RuntimeError(
                    f"TAPID decode request {i} scheduled {end - start} "
                    f"rows with {computed_list[i]} computed — chunked "
                    "prefill is unsupported"
                )
            slot = self._tapid_acquire_slot()
            prompt_len = end - start
            # Budget up to the region limit; vLLM finishes the request at its
            # own max_tokens (or eos) earlier and the slot is released then.
            budget = max(1, self._tapid_max_len - prompt_len)
            self.tapid_session.kv_alloc(slot, prompt_len, budget)
            # Review 1.3: [T, 1] F32 token ids — the entry Embed SOP does the
            # vocabulary gather on-device.
            self.tapid_session.submit(
                slot, input_ids[start:end].tolist(), prompt_len, 1,
                dtype=DTYPE_F32,
            )
            step_slots.append(slot)
            n_prefills += 1
        self._tapid_step_slots = step_slots
        if n_prefills:
            logger.info(
                "TAPID decode: submitted %d prefill(s) slots=%s rows=%d",
                n_prefills,
                [s for s in step_slots if s is not None],
                rows,
            )
        self._tapid_fill_pages()
        # execute_model asserts a Tensor return; nothing torch-side reads it.
        return torch.zeros(
            (input_ids.shape[0], self._tapid_hidden_size),
            dtype=self.model_config.dtype,
            device=self.device,
        )

    def _tapid_fill_pages(self) -> None:
        """Mirror vLLM's block allocation into the device's page table.

        The paged-KV device contract (gpu_daemon kv_cache_types.cuh) keeps
        the page table TAPID-owned; this filler feeds it the full-attention
        group's block ids each step so the engine owns the allocation
        decisions (the same ids, at the same 16-token granularity — the
        arm-time check below pins that). Rows are append-only within a
        generation, so a slot whose page count did not grow is skipped.
        """
        session = self.tapid_session
        if not hasattr(session, "kv_set_pages"):
            return  # shim predates the filler seam: built-in allocator rules
        input_batch = self._tapid_input_batch
        block_tables = getattr(input_batch, "block_table", None)
        if input_batch is None or block_tables is None:
            return
        fa_bt = block_tables[0]  # group 0 is the full-attention cache group
        if fa_bt.kv_cache_block_size != 16:
            raise RuntimeError(
                f"TAPID paged KV pins KV_PAGE_SIZE=16 but the FA cache "
                f"group runs block_size={fa_bt.kv_cache_block_size}"
            )
        for req_id, slot in self._tapid_req_slot.items():
            row = input_batch.req_id_to_index.get(req_id)
            if row is None:
                continue
            n = int(fa_bt.num_blocks_per_row[row])
            if n == 0 or self._tapid_slot_page_count.get(slot) == n:
                continue
            session.kv_set_pages(slot, fa_bt.block_table.np[row, :n].tolist())
            self._tapid_slot_page_count[slot] = n
            logger.debug(
                "TAPID decode: slot %d page row -> %d block(s) (req %s)",
                slot, n, req_id,
            )

    def _tapid_acquire_slot(self) -> int:
        if not self._tapid_slot_free:
            raise RuntimeError(
                "TAPID decode: no free slots (all "
                f"{len(self._tapid_req_slot)} occupied); raise --max-num-seqs "
                "limits or wait for requests to finish"
            )
        return self._tapid_slot_free.pop()

    def _tapid_release_slot(self, req_id: Any, slot: int) -> None:
        logger.info(
            "TAPID decode: releasing slot %d (req %s, %d tokens)",
            slot, req_id, self._tapid_generated.get(slot, 0),
        )
        self.tapid_session.kv_release(slot)
        self._tapid_req_slot.pop(req_id, None)
        self._tapid_slot_page_count.pop(slot, None)
        self._tapid_awaiting_first.discard(slot)
        self._tapid_done_slots.discard(slot)
        self._tapid_pending.pop(slot, None)
        self._tapid_generated.pop(slot, None)
        self._tapid_slot_free.append(slot)

    def _tapid_fetch_tokens(self, slot: int, *, first: bool) -> list[int]:
        """Drain one slot's terminal mailbox into token ids (col 1 of [T, 2])."""
        n_expect = self._tapid_max_len * 2 + 8
        deadline = time.monotonic() + (
            _TAPID_FETCH_TIMEOUT_MS if first else _TAPID_DECODE_TIMEOUT_MS
        ) / 1000.0
        chunk = _TAPID_FETCH_CHUNK_MS if first else _TAPID_DECODE_CHUNK_MS
        while True:
            try:
                payload, out_rows, out_cols, _mirrors = (
                    self.tapid_session.fetch_with_token_ids(
                        slot, n_expect, timeout_ms=chunk
                    )
                )
                break
            except self.tapid_door.TapidError as exc:
                if "timed out" not in str(exc):
                    raise
                now = time.monotonic()
                if now >= deadline:
                    logger.error(
                        "TAPID decode: slot %d produced no %s after %.0fs — "
                        "dumping device state",
                        slot, "first token" if first else "tokens",
                        now - deadline + (
                            _TAPID_FETCH_TIMEOUT_MS if first
                            else _TAPID_DECODE_TIMEOUT_MS
                        ) / 1000.0,
                    )
                    self._dump_tapid_stall(f"decode fetch slot={slot}")
                    raise
                logger.info(
                    "TAPID decode: still waiting for slot %d (%s)",
                    slot, "first token" if first else "next tokens",
                )
        if out_cols != 2:
            raise RuntimeError(
                f"TAPID decode: slot {slot} mailbox returned {out_cols} "
                "columns, expected the head SOP's [T, 2]"
            )
        # The head SOP's terminal [T, 2] carries one row per position: for a
        # prefill entry T == prompt rows (argmax at each position), for a
        # decode entry T == 1. Only the LAST row is the step's sampled token
        # — the same row the device daemon's done_token reads. Taking row 0
        # would report the position-0 argmax plus every prompt position's
        # argmax as "generated" tokens.
        return [int(payload[(out_rows - 1) * 2 + 1])]

    def _tapid_next_token(self, slot: int, *, first: bool) -> int:
        if slot in self._tapid_done_slots:
            # Device loop finished: nothing will ever be published again.
            # Async scheduling optimistically over-schedules one step past a
            # finish token (output placeholders); that ghost step must return
            # instantly or its fetch blocks the scheduler's finish
            # reconciliation until the fetch timeout. Reply with the same
            # finish token — the scheduler marks the stale output and drops
            # it.
            return self._tapid_eos_id
        buf = self._tapid_pending.get(slot)
        if buf:
            return buf.pop(0)
        tokens = self._tapid_fetch_tokens(slot, first=first)
        if not tokens:
            raise RuntimeError(
                f"TAPID decode: slot {slot} drained an empty mailbox entry"
            )
        head, *rest = tokens
        if rest:
            self._tapid_pending[slot] = rest
        return head

    def _decode_sample_tokens(self) -> Any:
        """Replace the sampler: report device-sampled tokens to the engine.

        Per engine step every scheduled request gets exactly one token: fresh
        prefills block for their head output (the prefill itself is the wait),
        decode rows pop the daemon's next token (backlog from earlier drains
        first, then a fresh fetch). Requests the scheduler finished have
        their slot released here — kv_release stops the device loop, and any
        tokens still buffered for them are discarded.
        """
        state = self.execute_model_state
        self.execute_model_state = None
        input_batch = state.input_batch
        req_ids = list(input_batch.req_ids)
        step_slots = self._tapid_step_slots
        self._tapid_step_slots = []
        if len(step_slots) != len(req_ids):
            raise RuntimeError(
                f"TAPID decode: step slot map has {len(step_slots)} entries "
                f"for {len(req_ids)} requests — forward/sampler desync"
            )
        for req_id, slot in zip(req_ids, step_slots):
            if slot is not None:
                if req_id in self._tapid_req_slot:
                    raise RuntimeError(
                        f"TAPID decode: request {req_id} prefilled twice"
                    )
                self._tapid_req_slot[req_id] = slot
                self._tapid_awaiting_first.add(slot)

        sampled: list[list[int]] = []
        for req_id, fresh_slot in zip(req_ids, step_slots):
            slot = fresh_slot
            first = False
            if slot is None:
                slot = self._tapid_req_slot.get(req_id)
            else:
                first = slot in self._tapid_awaiting_first
            if slot is None:
                raise RuntimeError(
                    f"TAPID decode: request {req_id} has no slot this step"
                )
            token = self._tapid_next_token(slot, first=first)
            if first:
                self._tapid_awaiting_first.discard(slot)
            if token == self._tapid_eos_id:
                # The daemon stops feeding on EOS — remember it so any
                # over-scheduled ghost step returns instantly (see
                # _tapid_next_token).
                self._tapid_done_slots.add(slot)
            self._tapid_generated[slot] = self._tapid_generated.get(slot, 0) + 1
            logger.info(
                "TAPID decode: slot %d token #%d = %d%s",
                slot, self._tapid_generated[slot], token,
                " (prefill)" if first else "",
            )
            sampled.append([token])

        for req_id in list(state.finished_req_ids or ()):
            slot = self._tapid_req_slot.get(req_id)
            if slot is not None:
                self._tapid_release_slot(req_id, slot)

        # NOTE: deliberately NOT calling postprocess_num_computed_tokens
        # here. Nothing reads req_states.num_computed_tokens.gpu on the
        # decode path (classification uses the slot map + num_computed_
        # tokens_np), and the first post-arm run of that postprocess kernel
        # has to JIT + cuModuleLoadData — a context-wide module load, which
        # deadlocks against the resident persistent kernel (observed on
        # cloud: hang right after the JIT warning). Never introduce a
        # host-side op post-arm that warmup did not already exercise.

        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={
                req_id: i for i, req_id in enumerate(req_ids)
            },
            sampled_token_ids=sampled,
            prompt_logprobs_dict={},
        )
        output.kv_connector_output = self.kv_connector.post_forward(
            state.finished_req_ids
        )
        return output

    def _dump_tapid_stall(self, where: str) -> None:
        """Device-side hang snapshot, twice: what did not move is the stall.

        tapid_debug_dump prints the live scheduler state (issuer status,
        scoreboards, work-pool budget) from inside the resident kernel's
        host agent. The dump must never mask the error that triggered it.
        """
        dump = getattr(self.tapid_session, "debug_dump", None)
        if dump is None:
            return
        for i in (1, 2):
            logger.error("TAPID stall snapshot #%d (%s):", i, where)
            try:
                dump(0)
            except Exception as exc:
                logger.error("TAPID debug_dump failed: %s", exc)
            if i == 1:
                time.sleep(_TAPID_DUMP_GAP_S)

    def _tapid_text_model(self) -> Any:
        candidate = getattr(self.model, "language_model", None) or self.model
        return getattr(candidate, "model", None) or candidate

    # ---- shutdown ----------------------------------------------------------

    def _close_tapid_session(self) -> None:
        self.tapid_armed = False
        if self.tapid_session is not None:
            # Best-effort slot teardown; close() would free the device state
            # anyway, but an explicit release keeps the audit log honest.
            for req_id, slot in list(self._tapid_req_slot.items()):
                try:
                    self._tapid_release_slot(req_id, slot)
                except Exception as exc:      # noqa: BLE001
                    logger.warning("TAPID: slot %d release failed: %s", slot, exc)
            self._tapid_slot_free = []
            # close() stops the persistent kernels, so a real device sync is
            # legal again — and vLLM's own shutdown path needs one.
            self.tapid_session.close()
            self.tapid_session = None
        _restore_device_sync()
        self._tapid_decoder_weights = None

    def shutdown(self) -> None:
        self._close_tapid_session()
        super().shutdown()
