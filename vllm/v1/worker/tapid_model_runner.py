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
* Prefill only. The program boundary is the decoder output; vLLM applies the
  final norm and samples. Decode, chunked prefill, and multi-request batches
  are refused with explicit errors — with the decoder weights freed there is
  no vLLM fallback either.

Model specifics (checkpoint conversion, program assembly, skeleton names) live
in the TAPID repo's model assembly package; nothing here knows the model.
"""

import importlib
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

logger = init_logger(__name__)

_TAPID_DOOR_MODULE = "tapid_vllm"
# Model-side half of the contract, on PYTHONPATH as $TAPID_REPO.
_TAPID_ADAPTER_MODULE = "models.model_assembly.qwen3_6_27b_dense.vllm_adapter"

# Weight upload of ~53 GiB and a set_program over the resident kernel can take
# minutes on a loaded host; nothing here should time out before they finish.
_TAPID_BIND_TIMEOUT_MS = 1_800_000
_TAPID_PROGRAM_TIMEOUT_MS = 600_000
_TAPID_FETCH_TIMEOUT_MS = 600_000

# Pre-arm the fetch can block far longer than a decode step ever would; a
# hung prefill surfaces as this timeout, not as a silent wedge.
_TAPID_PREFILL_LOG_EVERY = 8


def validate_tapid_config(runner: Any) -> None:
    tapid_config = runner.vllm_config.additional_config["tapid"]
    if tapid_config.get("model_signature") != "qwen3_5_dense_27b_bf16":
        raise ValueError("TAPID requires the Qwen3.5 27B BF16 signature")
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
    max_batched = int(runner.scheduler_config.max_num_batched_tokens)
    if max_batched > adapter.MAX_PREFILL_TOKENS:
        raise ValueError(
            "TAPID prefill takes the whole prompt in one step; "
            f"max_num_batched_tokens={max_batched} exceeds the kernel's row "
            f"buffer of {adapter.MAX_PREFILL_TOKENS}"
        )
    if runner.model_config.max_model_len > adapter.MAX_PREFILL_TOKENS:
        raise ValueError(
            f"max_model_len={runner.model_config.max_model_len} cannot be "
            f"prefilled in one step (kernel row buffer is "
            f"{adapter.MAX_PREFILL_TOKENS}); lower max_model_len"
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
        """Release every non-skeleton parameter back to the allocator.

        The persistent-kernel program is the only consumer of the decoder
        weights, so the vLLM copies are dead weight (~50 GiB). Params become
        empty meta tensors: any accidental access fails loudly instead of
        silently computing on dummy values.
        """
        skeleton_params = {
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        }
        freed_bytes = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in skeleton_params:
                    continue
                freed_bytes += param.numel() * param.element_size()
                param.data = torch.empty(0, dtype=param.dtype, device="meta")
        logger.info(
            "TAPID: freed %.1f GiB of decoder parameters from the vLLM model",
            freed_bytes / (1 << 30),
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
        logger.info(
            "TAPID: preloaded embed/stage/norm kernels for %d token counts",
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
        started = time.monotonic()
        self.tapid_session.launch()
        # Same settle window the bench runner uses: the resident CTAs come up
        # before the program is swapped in.
        time.sleep(0.3)
        payload = self.tapid_adapter.decoder_program_payload(
            self.tapid_session.abi
        )
        self.tapid_session.set_program(
            payload, timeout_ms=_TAPID_PROGRAM_TIMEOUT_MS
        )
        self.tapid_armed = True
        logger.info(
            "TAPID armed: persistent prefill program resident (%.1fs)",
            time.monotonic() - started,
        )

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

    def _prefill_batch(self) -> tuple[int, Any]:
        """Validate the scheduled batch is one fresh prefill; return (rows, md).

        Everything comes from the forward context's attention metadata as
        host-side reads (``query_start_loc.cpu()`` and ``.item()`` are plain
        memcpys) — no kernel launches, which are the dangerous ones once the
        persistent kernel is resident.
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
        if num_reqs != 1:
            raise RuntimeError(
                f"TAPID prefill runs exactly one request per step, got "
                f"{num_reqs}; run with --max-num-seqs 1 and one request at a "
                f"time"
            )
        rows = int(query_start_loc[-1].item())
        query_len = int(query_start_loc[1].item()) - int(query_start_loc[0].item())
        seq_len = int(batch_md.seq_lens[0].item())
        computed = seq_len - query_len
        if computed != 0:
            hint = (
                "decode step"
                if query_len == 1
                else "chunked prefill (run with --max-num-batched-tokens >= "
                "--max-model-len to prefill in one step)"
            )
            raise RuntimeError(
                f"TAPID prefill cannot continue a sequence ({hint}; "
                f"{computed} tokens already computed). Decode is out of scope: "
                f"use --max-tokens 1 to measure prefill only."
            )
        return rows, batch_md

    def _tapid_prefill_forward(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> Any:
        rows, _batch_md = self._prefill_batch()
        if rows <= 0 or rows > self.tapid_adapter.MAX_PREFILL_TOKENS:
            raise RuntimeError(
                f"TAPID prefill row count {rows} outside (0, "
                f"{self.tapid_adapter.MAX_PREFILL_TOKENS}]"
            )
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

        request_id = self._tapid_next_request
        self._tapid_next_request += 1
        self.tapid_session.submit_hidden(
            request_id, hidden[:rows], self._tapid_hidden_size
        )
        out_flat, out_rows, out_cols = self.tapid_session.fetch_array(
            request_id,
            rows * self._tapid_hidden_size,
            timeout_ms=_TAPID_FETCH_TIMEOUT_MS,
        )
        if (out_rows, out_cols) != (rows, self._tapid_hidden_size):
            raise RuntimeError(
                f"TAPID returned {out_rows}x{out_cols}, expected "
                f"{rows}x{self._tapid_hidden_size}"
            )

        # Writable numpy view over the fetch buffer -> device BF16 -> final
        # norm. Every module here was loaded by _warm_tapid_kernels; from_numpy
        # and the copies themselves launch no lazily-loaded kernels.
        out = torch.from_numpy(
            np.asarray(out_flat).reshape(out_rows, out_cols)
        ).to(self.device, dtype=self.model_config.dtype)
        normed = self._tapid_text_model().norm(out)
        if isinstance(normed, tuple):
            normed = normed[0]

        self._tapid_steps += 1
        if self._tapid_steps % _TAPID_PREFILL_LOG_EVERY == 1:
            logger.info(
                "TAPID prefill #%d: rows=%d request_id=%#x",
                self._tapid_steps, rows, request_id,
            )
        return normed

    def _tapid_text_model(self) -> Any:
        candidate = getattr(self.model, "language_model", None) or self.model
        return getattr(candidate, "model", None) or candidate

    # ---- shutdown ----------------------------------------------------------

    def _close_tapid_session(self) -> None:
        self.tapid_armed = False
        if self.tapid_session is not None:
            # close() stops the persistent kernels, so a real device sync is
            # legal again — and vLLM's own shutdown path needs one.
            self.tapid_session.close()
            self.tapid_session = None
        _restore_device_sync()
        self._tapid_decoder_weights = None

    def shutdown(self) -> None:
        self._close_tapid_session()
        super().shutdown()
