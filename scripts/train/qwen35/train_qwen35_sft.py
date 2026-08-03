#!/usr/bin/env python3
"""Train Qwen3.5 on the native-tool packed NumPy contract.

This entry point is separate from the OLMoCore trainer.  It requires a pinned
Transformers build with Qwen3.5 support and refuses to train unless packed
document isolation is available in full attention, linear attention, and the
causal convolution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import (
    HfArgumentParser,
    Qwen2TokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainerCallback,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers.models.qwen3_5 import modeling_qwen3_5

from open_instruct.qwen35_checkpoint_resume import load_qwen35_text_checkpoint_for_trainer
from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID as CHUNKED_LOSS_IMPLEMENTATION_ID
from open_instruct.qwen35_chunked_loss import QUALIFIED_CHUNK_SIZES, install_qwen35_checkpointed_chunked_loss
from open_instruct.qwen35_data import Qwen35NumpyPackedDataset, Qwen35PackedCollator
from open_instruct.qwen35_hardware import Qwen35CudaEventTimerCallback, Qwen35HardwareProfilerCallback
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_h4 import (
    H4_PREREGISTRATION_CLOSURE_SHA256,
    load_h4_contract,
    validate_forward_loss_audit,
)
from open_instruct.qwen35_qualification_r18_h4 import sha256_file as h4_sha256_file
from open_instruct.qwen35_reporting import (
    Qwen35FlopFormula,
    Qwen35WindowCounts,
    append_jsonl,
    build_reporting_record,
    summarize_reporting_records,
)
from open_instruct.qwen35_schedule import ScheduledQwen35Dataset, validate_schedule_manifest
from open_instruct.qwen35_training import (
    build_text_conversion_ledger,
    validate_fp32_optimizer_state,
    validate_fp32_trainable_parameters,
    validate_text_loading_info,
    write_json_atomic,
)

DEFAULT_MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PINNED_TRANSFORMERS_COMMIT = "d7d894cf917562d62c61497588ab64e4ae2c699d"
EXPECTED_SUITE_ID = "v3-semantic-causal-suite-r1-core-frozen"
EXPECTED_RENDERER_AMENDMENT_ID = "v3-semantic-causal-qwen35-32k-referenced-tool-pruning-r1"
EXPECTED_RENDERER_AMENDMENT_SHA256 = "2875ba0f0953c253a7676af4b460f9987157c312718e99e89d2916d85a5f84a0"
EXPECTED_CHAT_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
EXPECTED_CORE_OPERATIONS_SHA256 = "73feaeedf0644fbdc1ef4399b55a9b4c50e045f74c0b68f0afe695da2505f572"
EXPECTED_FROZEN_DESIGN_SHA256 = "8b24efdd66429280911fa983d46c5d0116ddfe72a41e1cf56965bacc6b6a3b94"
EXPECTED_AMENDED_SAMPLE_UIDS = {
    "cd3392225f3998340230426b48a40bc3a5918157fee30f06d5aa575e74ca45e4",
    "591028f79894dd50873a4c51a9fbe60d5bc35b910adf571418a1774d25bf47fd",
    "a0d39fcac3cb83b4987c3fc8b1638ab17d39c7043cd83a218833bd3b002a87a8",
    "7558c6b2233fc982bfb38af45651084c203b046a0ab124f84fdf0a68b960036f",
    "2556104fb4601af091cd5c7a1a08ed7464aac2266dbe27a026ed2eaf68f0f159",
}
PINNED_KERNEL_VERSIONS = {
    "flash-attn": "2.8.3",
    "causal-conv1d": "1.6.2.post1",
    "flash-linear-attention": "0.4.1",
    "fla-core": "0.4.1",
}
PINNED_RUNTIME_VERSIONS = {
    "accelerate": "1.12.0",
    "numpy": "2.5.1",
    "torch": "2.9.1+cu129",
    "torchvision": "0.24.1+cu129",
}


def distributed_barrier_on_local_cuda_device() -> None:
    """Synchronize an initialized NCCL job without rank-to-device inference."""

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.barrier(device_ids=[local_rank])


def destroy_initialized_process_group() -> None:
    """Shut down the default process group on normal and exceptional exits."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base"
    model_revision: str = DEFAULT_MODEL_REVISION
    cache_dir: str | None = None
    use_liger_fused_linear_cross_entropy: bool = False
    use_checkpointed_chunked_selected_loss: bool = True
    selected_loss_chunk_size: int = 256
    hash_text_conversion_tensors: bool = True


@dataclass
class DataArguments:
    numpy_data_dir: str = field(metadata={"help": "Directory produced by convert_sft_data_for_qwen35.py"})
    expected_arm_id: str = field(metadata={"help": "Frozen arm identity, C00 through C11"})
    pack_schedule_path: str = field(metadata={"help": "Frozen hashed no-repeat pack schedule JSON"})
    expected_schedule_sha256: str = field(metadata={"help": "Expected schedule_sha256 embedded in the schedule"})
    sequence_length: int = 32768
    verify_data_hashes: bool = False
    drop_last: bool = False


@dataclass
class Qwen35TrainingArguments(HFTrainingArguments):
    resume_from_checkpoint: str | None = None
    expected_initial_global_step: int = 0
    expected_final_global_step: int | None = None
    stop_after_steps: int | None = None
    exact_metrics_sync_interval: int = 1
    hardware_qualification_manifest: str | None = None
    h4_qualification_contract: str | None = None
    h4_preregistration_closure_sha256: str | None = None
    h4_assay: str | None = None
    hardware_profile: bool = False
    cuda_event_step_timing: bool = False
    require_no_dense_logits: bool = False
    require_forward_loss_audit: bool = False


def evenly_spaced_integer_indices(length: int, count: int) -> list[int]:
    """Return deterministic, in-bounds indices without floating-point rounding.

    The first and last indices are always selected when ``count > 1``.  Exact
    integer arithmetic is required here: CUDA integer ``torch.linspace`` can
    round the final index above ``length - 1`` for sufficiently large tensors.
    """

    if length <= 0:
        raise ValueError("integer-index population length must be positive")
    if count <= 0:
        raise ValueError("integer-index sample count must be positive")
    if count > length:
        raise ValueError("integer-index sample count cannot exceed population length")
    if count == 1:
        return [0]
    denominator = count - 1
    maximum = length - 1
    return [(position * maximum) // denominator for position in range(count)]


def parameter_probe_samples(
    model: torch.nn.Module, *, parameter_limit: int = 32, values_per_parameter: int = 64
) -> dict[str, dict[str, Any]]:
    """Capture deterministic sparse FP32 samples across trainable parameters."""

    if parameter_limit <= 0 or values_per_parameter <= 0:
        raise ValueError("parameter-probe limits must be positive")
    candidates = [(name, parameter) for name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    if not candidates:
        raise ValueError("model has no trainable parameter for the update probe")
    selected_count = min(parameter_limit, len(candidates))
    selected_positions = evenly_spaced_integer_indices(len(candidates), selected_count)
    samples: dict[str, dict[str, Any]] = {}
    for position in selected_positions:
        name, parameter = candidates[int(position)]
        flattened = parameter.detach().reshape(-1)
        value_count = min(values_per_parameter, flattened.numel())
        indices = torch.tensor(
            evenly_spaced_integer_indices(flattened.numel(), value_count),
            dtype=torch.int64,
            device=parameter.device,
        )
        values = flattened.index_select(0, indices).float().cpu().tolist()
        samples[name] = {"indices": indices.cpu().tolist(), "values": values}
    return samples


def compare_parameter_probe_samples(
    initial: dict[str, dict[str, Any]], final: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compare two sparse captures and fail on drift, non-finite values, or no update."""

    if initial.keys() != final.keys():
        raise ValueError("parameter-probe name set changed during training")
    changed_values = 0
    total_values = 0
    l1_delta = 0.0
    max_absolute_delta = 0.0
    initial_digest = hashlib.sha256()
    final_digest = hashlib.sha256()
    for name in initial:
        if initial[name]["indices"] != final[name]["indices"]:
            raise ValueError(f"parameter-probe indices changed for {name}")
        left_values = initial[name]["values"]
        right_values = final[name]["values"]
        if len(left_values) != len(right_values):
            raise ValueError(f"parameter-probe value count changed for {name}")
        for left, right in zip(left_values, right_values, strict=True):
            left = float(left)
            right = float(right)
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"parameter-probe captured a non-finite value in {name}")
            delta = abs(right - left)
            changed_values += delta > 0
            total_values += 1
            l1_delta += delta
            max_absolute_delta = max(max_absolute_delta, delta)
        identity = json.dumps({"name": name, "indices": initial[name]["indices"]}, separators=(",", ":")).encode()
        initial_digest.update(identity)
        final_digest.update(identity)
        initial_digest.update(json.dumps(left_values, separators=(",", ":")).encode())
        final_digest.update(json.dumps(right_values, separators=(",", ":")).encode())
    if changed_values == 0 or max_absolute_delta <= 0:
        raise RuntimeError("no sampled trainable parameter changed during training")
    return {
        "sampled_parameters": len(initial),
        "sampled_values": total_values,
        "changed_sampled_values": changed_values,
        "l1_delta": l1_delta,
        "max_absolute_delta": max_absolute_delta,
        "initial_values_sha256": initial_digest.hexdigest(),
        "final_values_sha256": final_digest.hexdigest(),
    }


class Qwen35UpdateProbeCallback(TrainerCallback):
    """Fail-closed finite-loss/gradient and nonzero-parameter-update proof."""

    def __init__(
        self, output_dir: Path, expected_initial_global_step: int, expected_final_global_step: int | None = None
    ) -> None:
        self.output_dir = output_dir
        self.expected_initial_global_step = expected_initial_global_step
        self.expected_final_global_step = expected_final_global_step
        self.observed_initial_global_step: int | None = None
        self.initial: dict[str, dict[str, Any]] | None = None
        self.losses: list[dict[str, float | int]] = []
        self.grad_norms: list[dict[str, float | int]] = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step != self.expected_initial_global_step:
            raise RuntimeError(
                f"loaded initial global step {state.global_step} != expected {self.expected_initial_global_step}"
            )
        self.observed_initial_global_step = int(state.global_step)
        self.initial = parameter_probe_samples(model)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        for key, target in (("loss", self.losses), ("grad_norm", self.grad_norms)):
            if key not in logs:
                continue
            value = float(logs[key])
            if not math.isfinite(value):
                raise RuntimeError(f"training emitted non-finite {key} at step {state.global_step}")
            target.append({"step": int(state.global_step), "value": value})

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.initial is None:
            raise RuntimeError("parameter probe did not capture the train-begin model")
        if state.global_step <= self.expected_initial_global_step:
            raise RuntimeError("training completed without an optimizer step")
        expected_final = (
            state.max_steps if self.expected_final_global_step is None else self.expected_final_global_step
        )
        if state.global_step != expected_final:
            raise RuntimeError(
                f"training ended at global step {state.global_step}, expected final step {expected_final}"
            )
        if not self.losses or not self.grad_norms:
            raise RuntimeError("training emitted no finite loss or gradient-norm evidence")
        final_samples = parameter_probe_samples(model)
        comparison = compare_parameter_probe_samples(self.initial, final_samples)
        report = {
            "artifact": "qwen35_parameter_update_probe",
            "schema_version": 1,
            "status": "passed",
            "expected_initial_global_step": self.expected_initial_global_step,
            "observed_initial_global_step": self.observed_initial_global_step,
            "final_global_step": int(state.global_step),
            "optimizer_steps_observed": int(state.global_step - self.expected_initial_global_step),
            "finite_losses": self.losses,
            "finite_gradient_norms": self.grad_norms,
            "parameter_comparison": comparison,
            "initial_samples": self.initial,
            "final_samples": final_samples,
        }
        if state.is_world_process_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output = self.output_dir / "qwen35_parameter_update_probe.json"
            temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, output)


class Qwen35ExactMetricsCallback(TrainerCallback):
    """Audit every scheduled pack and report synchronized end-to-end windows."""

    METADATA_KEYS = (
        "_qwen35_schedule_index",
        "_qwen35_pack_index",
        "_qwen35_pack_uid",
        "_qwen35_synthetic",
        "_qwen35_real_tokens",
        "_qwen35_assistant_targets",
        "_qwen35_padding_tokens",
        "_qwen35_attention_length_squared",
        "_qwen35_document_count",
    )

    def __init__(
        self,
        *,
        output_dir: Path,
        sequence_length: int,
        schedule_sha256: str,
        formula: Qwen35FlopFormula,
        expected_initial_global_step: int,
        expected_final_global_step: int,
        sync_interval: int,
    ) -> None:
        if sync_interval <= 0:
            raise ValueError("exact metrics sync interval must be positive")
        if (expected_final_global_step - expected_initial_global_step) % sync_interval:
            raise ValueError("the requested step range must contain complete exact-metrics windows")
        self.output_dir = output_dir
        self.metrics_path = output_dir / "qwen35_exact_metrics.jsonl"
        self.summary_path = output_dir / "qwen35_exact_metrics_summary.json"
        self.sequence_length = sequence_length
        self.schedule_sha256 = schedule_sha256
        self.formula = formula
        self.expected_initial_global_step = expected_initial_global_step
        self.expected_final_global_step = expected_final_global_step
        self.sync_interval = sync_interval
        self.window_start_time: float | None = None
        self.window_counts = Qwen35WindowCounts()
        self.window_pack_uids: list[str] = []
        self.window_schedule_indices: list[int] = []
        self.window_loss_numerators: list[torch.Tensor] = []
        self.window_microbatch_divisors: list[torch.Tensor] = []
        self.window_selected_output_audits: list[dict[str, Any]] = []
        self.host_timings = {"data_wait": 0.0, "forward_loss": 0.0, "training_step": 0.0, "optimizer": 0.0}
        self.current_micro_step = 0
        self.data_fetch_started = False
        self.optimizer_started: float | None = None
        self.window_applied_learning_rates: list[float] = []
        self.window_gradient_dtype_audits: list[dict[str, Any]] = []
        self._world_size = 1
        self._process_index = 0
        self._require_forward_loss_audit = False
        self._selected_loss_chunk_size: int | None = None
        self._expected_vocabulary_size: int | None = None
        self._expected_hidden_size: int | None = None

    @staticmethod
    def _distributed() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @classmethod
    def _synchronize_window_boundary(cls) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if cls._distributed():
            distributed_barrier_on_local_cuda_device()

    def _window_offset(self, global_step: int) -> int:
        return global_step - self.expected_initial_global_step

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step != self.expected_initial_global_step:
            raise RuntimeError("exact metrics observed an unexpected initial global step")
        self._world_size = int(args.world_size)
        self._process_index = torch.distributed.get_rank() if self._distributed() else 0
        if self.metrics_path.exists():
            records = [json.loads(line) for line in self.metrics_path.read_text().splitlines() if line.strip()]
            summary = summarize_reporting_records(records)
            if summary["schedule_sha256"] != self.schedule_sha256:
                raise RuntimeError("existing exact metrics use a different schedule")
            if summary["last_step"] != self.expected_initial_global_step:
                raise RuntimeError("existing exact metrics do not end at the requested resume step")
        elif self.expected_initial_global_step == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        return control

    def begin_batch_fetch(self, global_step: int) -> None:
        if self._window_offset(global_step) % self.sync_interval == 0:
            if self.window_start_time is not None:
                raise RuntimeError("exact metrics attempted to overlap reporting windows")
            self._synchronize_window_boundary()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.window_start_time = time.perf_counter()
        self.data_fetch_started = True

    def record_data_wait(self, elapsed_seconds: float) -> None:
        if not self.data_fetch_started:
            raise RuntimeError("data wait ended without a matching batch-fetch start")
        self.host_timings["data_wait"] += elapsed_seconds
        self.data_fetch_started = False

    def on_step_begin(self, args, state, control, **kwargs):
        if self.window_start_time is None:
            raise RuntimeError("optimizer step began outside an exact-metrics window")
        self.current_micro_step = 0
        return control

    def expected_schedule_index(self, global_step: int, micro_step: int) -> int:
        local_microbatch = global_step * int(self._gradient_accumulation_steps) + micro_step
        return local_microbatch * self._world_size + self._process_index

    def configure_gradient_accumulation(self, gradient_accumulation_steps: int) -> None:
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient accumulation must be positive")
        self._gradient_accumulation_steps = gradient_accumulation_steps

    def configure_forward_loss_audit(
        self,
        *,
        required: bool,
        chunk_size: int,
        vocabulary_size: int,
        hidden_size: int,
    ) -> None:
        if chunk_size not in QUALIFIED_CHUNK_SIZES:
            raise ValueError("forward-loss audit chunk size is not qualified")
        if vocabulary_size <= 0 or hidden_size <= 0:
            raise ValueError("forward-loss audit geometry must be positive")
        self._require_forward_loss_audit = bool(required)
        self._selected_loss_chunk_size = int(chunk_size)
        self._expected_vocabulary_size = int(vocabulary_size)
        self._expected_hidden_size = int(hidden_size)

    def record_forward_loss_time(self, elapsed_seconds: float) -> None:
        self.host_timings["forward_loss"] += elapsed_seconds

    def record_microbatch(
        self,
        *,
        global_step: int,
        metadata_row: dict[str, Any],
        observed_assistant_targets: int,
        loss: torch.Tensor,
        num_items_in_batch: torch.Tensor | int | None,
        elapsed_seconds: float,
        loss_audit: dict[str, Any] | None = None,
    ) -> None:
        if num_items_in_batch is None:
            raise RuntimeError("Trainer did not provide the global assistant-target divisor")
        expected_index = self.expected_schedule_index(global_step, self.current_micro_step)
        observed_index = int(metadata_row["_qwen35_schedule_index"])
        if observed_index != expected_index:
            raise RuntimeError(f"schedule exposure drift: observed {observed_index}, expected {expected_index}")
        expected_targets = int(metadata_row["_qwen35_assistant_targets"])
        if observed_assistant_targets != expected_targets:
            raise RuntimeError(
                f"selective-row count {observed_assistant_targets} != scheduled target count {expected_targets}"
            )
        if self._require_forward_loss_audit and loss.dtype != torch.float32:
            raise RuntimeError(f"H4 selected-output loss must be FP32, found {loss.dtype}")
        divisor_integer = int(torch.as_tensor(num_items_in_batch).detach().reshape(()).cpu())
        if loss_audit is None:
            if self._require_forward_loss_audit:
                raise RuntimeError("selected-output forward emitted no loss audit")
        else:
            if self._selected_loss_chunk_size is None:
                raise RuntimeError("selected-output audit validation was not configured")
            validation = validate_forward_loss_audit(
                loss_audit,
                expected_selected_rows=observed_assistant_targets,
                expected_global_target_count=divisor_integer,
                expected_chunk_size=self._selected_loss_chunk_size,
                expected_vocabulary_size=int(self._expected_vocabulary_size),
                expected_hidden_size=int(self._expected_hidden_size),
            )
            self.window_selected_output_audits.append(
                {
                    "audit": loss_audit,
                    "loss_dtype": str(loss.dtype),
                    "pack_uid": str(metadata_row["_qwen35_pack_uid"]),
                    "schedule_index": observed_index,
                    "validation": validation,
                }
            )
        self.window_counts.add_pack(
            {
                "real_tokens": metadata_row["_qwen35_real_tokens"],
                "assistant_targets": expected_targets,
                "padding_tokens": metadata_row["_qwen35_padding_tokens"],
                "attention_length_squared": metadata_row["_qwen35_attention_length_squared"],
                "document_count": metadata_row["_qwen35_document_count"],
                "synthetic": metadata_row["_qwen35_synthetic"],
            },
            self.sequence_length,
        )
        self.window_schedule_indices.append(observed_index)
        self.window_pack_uids.append(str(metadata_row["_qwen35_pack_uid"]))
        divisor = torch.as_tensor(num_items_in_batch, device=loss.device).detach().reshape(())
        self.window_microbatch_divisors.append(divisor)
        # Trainer multiplies the globally normalized local loss by world size
        # before DDP averages gradients. Undo only that scale to recover this
        # rank's unnormalized numerator for exact cross-rank aggregation.
        self.window_loss_numerators.append(loss.detach().float() * divisor / self._world_size)
        self.host_timings["training_step"] += elapsed_seconds
        self.current_micro_step += 1

    def on_pre_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        if optimizer is None:
            raise RuntimeError("optimizer-step callback did not receive the optimizer")
        learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
        if len(learning_rates) != 1:
            raise RuntimeError(f"optimizer parameter groups use different learning rates: {sorted(learning_rates)}")
        learning_rate = learning_rates.pop()
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise RuntimeError(f"optimizer has an invalid learning rate: {learning_rate}")
        self.window_applied_learning_rates.append(learning_rate)
        if self._require_forward_loss_audit:
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            gradients = [parameter.grad for parameter in parameters]
            missing = sum(gradient is None for gradient in gradients)
            dtype_counts: dict[str, int] = {}
            gradient_numel = 0
            for gradient in gradients:
                if gradient is None:
                    continue
                key = str(gradient.dtype)
                dtype_counts[key] = dtype_counts.get(key, 0) + 1
                gradient_numel += gradient.numel()
            if missing or dtype_counts != {"torch.float32": len(gradients)}:
                raise RuntimeError(
                    f"H4 gradients are missing or non-FP32: missing={missing}, dtype_counts={dtype_counts}"
                )
            self.window_gradient_dtype_audits.append(
                {
                    "gradient_numel": gradient_numel,
                    "gradient_tensor_count": len(gradients),
                    "gradient_dtype_counts": dtype_counts,
                    "missing_gradient_count": missing,
                }
            )
        self.optimizer_started = time.perf_counter()
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.optimizer_started is None:
            raise RuntimeError("optimizer completed without a start timestamp")
        self.host_timings["optimizer"] += time.perf_counter() - self.optimizer_started
        self.optimizer_started = None
        return control

    def _reduce_counts(self, device: torch.device) -> Qwen35WindowCounts:
        names = list(vars(self.window_counts))
        values = torch.tensor([getattr(self.window_counts, name) for name in names], dtype=torch.int64, device=device)
        if self._distributed():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        return Qwen35WindowCounts(**dict(zip(names, (int(value) for value in values.cpu()), strict=True)))

    def _gather_pack_identities(self) -> tuple[list[int], list[str]]:
        local = list(zip(self.window_schedule_indices, self.window_pack_uids, strict=True))
        if self._distributed():
            gathered: list[list[tuple[int, str]] | None] = [None] * self._world_size
            torch.distributed.all_gather_object(gathered, local)
            combined = [row for rank_rows in gathered for row in (rank_rows or [])]
        else:
            combined = local
        combined.sort()
        return [row[0] for row in combined], [row[1] for row in combined]

    def _gather_selected_output_audits(self) -> list[dict[str, Any]]:
        local = list(self.window_selected_output_audits)
        if self._distributed():
            gathered: list[list[dict[str, Any]] | None] = [None] * self._world_size
            torch.distributed.all_gather_object(gathered, local)
            combined = [row for rank_rows in gathered for row in (rank_rows or [])]
        else:
            combined = local
        combined.sort(key=lambda row: int(row["schedule_index"]))
        return combined

    def _reduce_host_timings(self, device: torch.device) -> dict[str, float]:
        names = list(self.host_timings)
        local = torch.tensor([self.host_timings[name] for name in names], dtype=torch.float64, device=device)
        maximum = local.clone()
        total = local.clone()
        if self._distributed():
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        result = {}
        for index, name in enumerate(names):
            result[f"{name}_rank_max"] = float(maximum[index].cpu())
            result[f"{name}_rank_mean"] = float(total[index].cpu()) / self._world_size
        return result

    def _finish_window(self, args, state, optimizer) -> None:
        if self.window_start_time is None:
            raise RuntimeError("exact metrics ended a window that never started")
        if self.current_micro_step != int(args.gradient_accumulation_steps):
            raise RuntimeError("optimizer step did not consume the configured number of microbatches")
        self._synchronize_window_boundary()
        elapsed = time.perf_counter() - self.window_start_time
        device = args.device
        counts = self._reduce_counts(device)
        schedule_indices, pack_uids = self._gather_pack_identities()
        selected_output_audits = self._gather_selected_output_audits()
        if self._require_forward_loss_audit and len(selected_output_audits) != counts.packs:
            raise RuntimeError(
                f"selected-output audit count {len(selected_output_audits)} != global pack count {counts.packs}"
            )
        if selected_output_audits and [row["schedule_index"] for row in selected_output_audits] != schedule_indices:
            raise RuntimeError("selected-output audit schedule identities drifted from exact pack accounting")
        if selected_output_audits and [row["pack_uid"] for row in selected_output_audits] != pack_uids:
            raise RuntimeError("selected-output audit pack identities drifted from exact pack accounting")

        microbatch_divisors = [int(value.cpu()) for value in self.window_microbatch_divisors]
        accumulation = int(args.gradient_accumulation_steps)
        expected_divisor = 0
        for offset in range(0, len(microbatch_divisors), accumulation):
            group = microbatch_divisors[offset : offset + accumulation]
            if len(group) != accumulation or len(set(group)) != 1:
                raise RuntimeError("global target divisor drifted within an optimizer step")
            expected_divisor += group[0]
        if expected_divisor != counts.assistant_targets:
            raise RuntimeError(
                f"global target divisors sum to {expected_divisor}, but schedule accounting gives "
                f"{counts.assistant_targets}"
            )
        loss_numerator = torch.stack(self.window_loss_numerators).sum()
        if self._distributed():
            torch.distributed.all_reduce(loss_numerator, op=torch.distributed.ReduceOp.SUM)
        normalized_loss = float(loss_numerator.cpu()) / counts.assistant_targets
        if not math.isfinite(normalized_loss):
            raise RuntimeError("exact metrics reconstructed a non-finite normalized loss")

        peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        if torch.cuda.is_available() and self._distributed():
            memory = torch.tensor([peak_allocated, peak_reserved], dtype=torch.int64, device=device)
            torch.distributed.all_reduce(memory, op=torch.distributed.ReduceOp.MAX)
            peak_allocated, peak_reserved = (int(value) for value in memory.cpu())
        host_timing = self._reduce_host_timings(device)
        validate_fp32_optimizer_state(optimizer, require_initialized=True)
        if len(self.window_applied_learning_rates) != self.sync_interval:
            raise RuntimeError("applied learning-rate accounting does not match the reporting window")
        learning_rate = self.window_applied_learning_rates[-1]
        record = build_reporting_record(
            formula=self.formula,
            step=int(state.global_step),
            world_size=self._world_size,
            elapsed_seconds=elapsed,
            counts=counts,
            schedule_sha256=self.schedule_sha256,
            pack_uids=pack_uids,
            schedule_indices=schedule_indices,
            learning_rate=learning_rate,
            normalized_loss=normalized_loss,
            global_target_divisor=counts.assistant_targets,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            synchronized=True,
            host_timing_seconds=host_timing,
            optimizer_updates=self.sync_interval,
            applied_learning_rates=self.window_applied_learning_rates,
        )
        record["selected_output_audits"] = selected_output_audits
        record["gradient_dtype_audits"] = list(self.window_gradient_dtype_audits)
        if state.is_world_process_zero:
            append_jsonl(self.metrics_path, record)
        self.window_start_time = None
        self.window_counts = Qwen35WindowCounts()
        self.window_pack_uids.clear()
        self.window_schedule_indices.clear()
        self.window_loss_numerators.clear()
        self.window_microbatch_divisors.clear()
        self.window_selected_output_audits.clear()
        self.window_applied_learning_rates.clear()
        self.window_gradient_dtype_audits.clear()
        self.host_timings = {key: 0.0 for key in self.host_timings}

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        if self._window_offset(state.global_step) % self.sync_interval == 0:
            self._finish_window(args, state, optimizer)
        if state.global_step >= self.expected_final_global_step:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.global_step != self.expected_final_global_step:
            raise RuntimeError(
                f"exact metrics ended at step {state.global_step}, expected {self.expected_final_global_step}"
            )
        if self.window_start_time is not None or self.window_counts.packs:
            raise RuntimeError("training ended with an incomplete exact-metrics window")
        if state.is_world_process_zero:
            loaded_liger_modules = sorted(
                name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel.")
            )
            if self._require_forward_loss_audit and loaded_liger_modules:
                raise RuntimeError(f"R18 H4 process imported forbidden Liger modules: {loaded_liger_modules[:10]}")
            records = [json.loads(line) for line in self.metrics_path.read_text().splitlines() if line.strip()]
            summary = summarize_reporting_records(records)
            audits = [row for record in records for row in record.get("selected_output_audits", [])]
            summary["selected_output_audit_count"] = len(audits)
            summary["selected_output_audits_sha256"] = hashlib.sha256(
                json.dumps(audits, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            gradient_audits = [row for record in records for row in record.get("gradient_dtype_audits", [])]
            summary["gradient_dtype_audit_count"] = len(gradient_audits)
            summary["gradient_dtype_audits_sha256"] = hashlib.sha256(
                json.dumps(gradient_audits, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            summary["loaded_liger_modules"] = loaded_liger_modules
            write_json_atomic(self.summary_path, summary)
        return control


class Qwen35ExactTrainer(Trainer):
    """Trainer adapter that strips audit metadata and proves schedule/loss invariants."""

    def __init__(self, *args, exact_metrics: Qwen35ExactMetricsCallback, **kwargs) -> None:
        self.exact_metrics = exact_metrics
        super().__init__(*args, **kwargs)
        if not self.model_accepts_loss_kwargs:
            raise RuntimeError("Trainer did not recognize the patched model's global loss-divisor kwargs")
        if not self._loss_shifts_labels:
            raise RuntimeError("Trainer did not recognize Qwen3.5 as a causal next-token loss")
        self.accelerator.even_batches = False
        if self.accelerator.even_batches:
            raise RuntimeError("Accelerate refused to disable duplicate tail batches")
        self.exact_metrics.configure_gradient_accumulation(int(self.args.gradient_accumulation_steps))
        self.exact_metrics.configure_forward_loss_audit(
            required=bool(self.args.require_forward_loss_audit),
            chunk_size=int(self.model._qwen35_selected_loss_chunk_size),
            vocabulary_size=int(self.model.config.vocab_size),
            hidden_size=int(self.model.config.hidden_size),
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore Qwen3.5 weights without Trainer's unsafe ``strict=False`` path."""

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            raise RuntimeError("strict Qwen3.5 Trainer resume does not support FSDP or DeepSpeed")
        target_model = self.model if model is None else model
        if target_model is not self.model:
            raise RuntimeError("strict Qwen3.5 Trainer resume requires the original live model object")
        rank = int(self.args.process_index)
        output_dir = Path(self.args.output_dir)
        audit_path = output_dir / f"qwen35_resume_model_load_audit_rank_{rank}.json"
        failure_path = output_dir / f"qwen35_resume_model_load_failure_rank_{rank}.json"
        if audit_path.exists() or failure_path.exists():
            raise RuntimeError(f"stale rank-{rank} resume-load evidence already exists")

        audit = None
        local_error = None
        try:
            audit = load_qwen35_text_checkpoint_for_trainer(target_model, resume_from_checkpoint)
        except Exception as error:
            local_error = error

        parameter = next(target_model.parameters())
        local_success = torch.tensor(
            [0 if local_error is not None else 1], dtype=torch.int32, device=parameter.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_success, op=torch.distributed.ReduceOp.MIN)
        globally_passed = int(local_success.item()) == 1
        if local_error is not None or not globally_passed:
            write_json_atomic(
                failure_path,
                {
                    "artifact": "qwen35_strict_trainer_checkpoint_load_failure",
                    "schema_version": 1,
                    "status": "failed",
                    "rank": rank,
                    "checkpoint_dir": str(Path(resume_from_checkpoint).resolve()),
                    "local_error_type": None if local_error is None else type(local_error).__name__,
                    "local_error": None if local_error is None else str(local_error),
                    "all_ranks_passed": globally_passed,
                    "optimizer_execution_authorized": False,
                },
            )
            if local_error is not None:
                raise RuntimeError("strict Qwen3.5 checkpoint restoration failed") from local_error
            raise RuntimeError("another rank failed strict Qwen3.5 checkpoint restoration")

        if audit is None:
            raise RuntimeError("strict Qwen3.5 checkpoint loader returned no audit")
        audit["rank"] = rank
        audit["all_ranks_passed"] = True
        audit["optimizer_execution_authorized"] = True
        write_json_atomic(audit_path, audit)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        self.exact_metrics.begin_batch_fetch(int(self.state.global_step))
        started = time.perf_counter()
        result = super().get_batch_samples(epoch_iterator, num_batches, device)
        self.exact_metrics.record_data_wait(time.perf_counter() - started)
        return result

    def create_optimizer(self, model=None):
        validate_fp32_trainable_parameters(self.model if model is None else model)
        optimizer = super().create_optimizer(model=model)
        validate_fp32_optimizer_state(optimizer, require_initialized=False)
        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        started = time.perf_counter()
        if self.args.require_no_dense_logits:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            logits = getattr(outputs, "logits", None)
            if logits is not None and logits.numel() != 0:
                raise RuntimeError(f"qualification path materialized dense logits with shape {tuple(logits.shape)}")
            result = (loss, outputs) if return_outputs else loss
        else:
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        observed_loss = result[0] if isinstance(result, tuple) else result
        if not torch.is_tensor(observed_loss) or observed_loss.ndim != 0 or not bool(torch.isfinite(observed_loss)):
            raise RuntimeError("qualification path produced a missing, nonscalar, or non-finite loss")
        self.exact_metrics.record_forward_loss_time(time.perf_counter() - started)
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        metadata_row = {}
        for key in self.exact_metrics.METADATA_KEYS:
            if key not in inputs:
                raise RuntimeError(f"scheduled training batch is missing {key}")
            metadata_row[key] = inputs.pop(key)
        observed_targets = int(inputs["shift_labels"].ne(-100).sum())
        started = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        audit_owner = self.accelerator.unwrap_model(model)
        loss_audit = getattr(audit_owner, "_qwen35_last_loss_audit", None)
        if loss_audit is not None:
            loss_audit = json.loads(json.dumps(loss_audit, allow_nan=False))
        self.exact_metrics.record_microbatch(
            global_step=int(self.state.global_step),
            metadata_row=metadata_row,
            observed_assistant_targets=observed_targets,
            loss=loss,
            num_items_in_batch=num_items_in_batch,
            loss_audit=loss_audit,
            elapsed_seconds=time.perf_counter() - started,
        )
        return loss

    def floating_point_ops(self, inputs):
        # Transformers' generic decoder estimate does not understand Qwen3.5's
        # hybrid GDN/full-attention architecture, isolated document lengths, or
        # selected output rows. Keep its ambiguous `total_flos` at zero; the
        # versioned qwen35_exact_metrics stream is authoritative.
        return 0


def verify_runtime() -> dict[str, str]:
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration") or not hasattr(transformers, "Qwen3_5ForCausalLM"):
        raise RuntimeError("installed Transformers lacks a required Qwen3.5 class; install requirements-qwen35.txt")
    source_version = getattr(transformers, "__version__", "unknown")
    if "dev" not in source_version and not source_version.startswith("5."):
        raise RuntimeError(
            f"Qwen3.5 training requires the pinned Transformers 5.x source build, found {source_version}"
        )
    direct_url_text = metadata.distribution("transformers").read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("Transformers installation has no direct_url.json; cannot prove the pinned source commit")
    direct_url = json.loads(direct_url_text)
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    source_url = direct_url.get("url", "")
    archive_pinned = (
        isinstance(source_url, str) and PINNED_TRANSFORMERS_COMMIT in source_url and "/archive/" in source_url
    )
    if installed_commit != PINNED_TRANSFORMERS_COMMIT and not archive_pinned:
        raise RuntimeError(
            f"Transformers source commit mismatch: {installed_commit!r} != {PINNED_TRANSFORMERS_COMMIT!r}"
        )
    installed = {"transformers_commit": PINNED_TRANSFORMERS_COMMIT, "transformers_source_url": str(source_url)}
    for package, expected_version in {**PINNED_RUNTIME_VERSIONS, **PINNED_KERNEL_VERSIONS}.items():
        installed_version = metadata.version(package)
        if installed_version != expected_version:
            raise RuntimeError(f"{package} version mismatch: {installed_version!r} != {expected_version!r}")
        installed[package] = installed_version
    return installed


def verify_packed_kernel_support(model: Qwen3_5ForCausalLM) -> None:
    if model.config._attn_implementation != "flash_attention_2":
        raise RuntimeError("packed Qwen3.5 training requires attn_implementation=flash_attention_2")
    if modeling_qwen3_5.chunk_gated_delta_rule is None:
        raise RuntimeError("flash-linear-attention is required so linear layers honor cu_seq_lens_q")
    linear_layers = [module for module in model.modules() if type(module).__name__ == "Qwen3_5GatedDeltaNet"]
    if not linear_layers:
        raise RuntimeError("Qwen3.5 model contains no expected gated-delta linear-attention layers")
    missing_causal_conv = [index for index, module in enumerate(linear_layers) if module.causal_conv1d_fn is None]
    if missing_causal_conv:
        raise RuntimeError(
            "causal-conv1d with seq_idx support is required for packed boundary isolation; "
            f"missing on {len(missing_causal_conv)} layers"
        )


def validate_arguments(
    model_args: ModelArguments, data_args: DataArguments, training_args: HFTrainingArguments
) -> None:
    if model_args.use_liger_fused_linear_cross_entropy:
        raise ValueError("Liger was abandoned after the independently validated R17 matched-reference failure")
    if not model_args.use_checkpointed_chunked_selected_loss:
        raise ValueError("the R18 production path requires checkpointed chunked selected-row loss")
    if model_args.selected_loss_chunk_size not in QUALIFIED_CHUNK_SIZES:
        raise ValueError(f"--selected_loss_chunk_size must be one of {QUALIFIED_CHUNK_SIZES}")
    if data_args.sequence_length <= 0:
        raise ValueError("--sequence_length must be positive")
    if data_args.sequence_length != 32768:
        raise ValueError("the frozen Qwen3.5 causal suite requires --sequence_length 32768")
    if data_args.expected_arm_id not in {f"C{index:02d}" for index in range(12)}:
        raise ValueError("--expected_arm_id must be one of C00 through C11")
    if data_args.drop_last:
        raise ValueError("the frozen no-repeat compute policy requires --drop_last false")
    if not data_args.verify_data_hashes:
        raise ValueError("the frozen Qwen3.5 trainer requires --verify_data_hashes true")
    if training_args.per_device_train_batch_size != 1:
        raise ValueError("pre-packed NumPy data requires --per_device_train_batch_size 1")
    if training_args.remove_unused_columns:
        raise ValueError("set --remove_unused_columns false so seq_idx and cu_seq_lens reach Qwen3.5")
    if not training_args.bf16:
        raise ValueError("the preregistered Qwen3.5 path requires --bf16 true")
    if training_args.fp16:
        raise ValueError("do not combine --fp16 with the required BF16 path")
    if getattr(training_args, "expected_initial_global_step", -1) < 0:
        raise ValueError("--expected_initial_global_step cannot be negative")
    if not data_args.expected_schedule_sha256 or len(data_args.expected_schedule_sha256) != 64:
        raise ValueError("--expected_schedule_sha256 must be a 64-character digest")
    if training_args.max_steps <= 0:
        raise ValueError("the frozen schedule path requires a positive --max_steps")
    expected_final = training_args.expected_final_global_step
    if expected_final is None:
        expected_final = training_args.stop_after_steps or training_args.max_steps
    if not training_args.expected_initial_global_step < expected_final <= training_args.max_steps:
        raise ValueError("expected final step must be after the initial step and at most max_steps")
    if training_args.stop_after_steps is not None and training_args.stop_after_steps != expected_final:
        raise ValueError("--stop_after_steps and --expected_final_global_step disagree")
    if not training_args.average_tokens_across_devices:
        raise ValueError("global assistant-target normalization requires --average_tokens_across_devices true")
    strategy = getattr(training_args.train_sampling_strategy, "value", training_args.train_sampling_strategy)
    if strategy != "sequential":
        raise ValueError("the hashed pack schedule requires --train_sampling_strategy sequential")
    if training_args.data_seed != training_args.seed:
        raise ValueError("paired runs require --data_seed to equal the nominal training/schedule seed")
    if training_args.dataloader_drop_last:
        raise ValueError("do not drop a scheduled distributed batch")
    if training_args.ignore_data_skip:
        raise ValueError("resume correctness requires --ignore_data_skip false")
    if not training_args.gradient_checkpointing:
        raise ValueError("32K Qwen3.5 qualification requires --gradient_checkpointing true")
    if training_args.use_liger_kernel:
        raise ValueError("disable generic --use_liger_kernel; R18 forbids every Liger execution path")
    optimizer_name = getattr(training_args.optim, "value", training_args.optim)
    if optimizer_name not in {"adamw_torch", "adamw_torch_fused"}:
        raise ValueError("the FP32 optimizer-state contract requires AdamW torch or AdamW torch fused")
    global_fixed_tokens = (
        data_args.sequence_length
        * training_args.world_size
        * training_args.gradient_accumulation_steps
        * training_args.per_device_train_batch_size
    )
    if global_fixed_tokens != 262_144:
        raise ValueError(f"frozen global fixed-token budget is 262144, found {global_fixed_tokens}")
    if (
        training_args.hardware_profile
        or training_args.cuda_event_step_timing
        or training_args.require_no_dense_logits
        or training_args.require_forward_loss_audit
        or training_args.h4_qualification_contract
    ):
        if not training_args.hardware_qualification_manifest:
            raise ValueError("hardware qualification flags require --hardware_qualification_manifest")
        qualification, qualification_sha256 = load_qualification_manifest(
            Path(training_args.hardware_qualification_manifest)
        )
        if qualification["protocol_id"] != "qwen35-hardware-qualification-r18":
            raise ValueError("the non-Liger production path requires the R18 qualification protocol")
        h2 = qualification["h2_acceptance"]
        if h2["production_implementation_id"] != CHUNKED_LOSS_IMPLEMENTATION_ID:
            raise ValueError("qualification manifest and production selected-loss implementation disagree")
        if model_args.selected_loss_chunk_size not in h2["candidate_chunk_sizes"]:
            raise ValueError("selected chunk size is absent from the R18 qualification candidate set")
        if qualification["runtime_pins"].get("liger_execution_allowed") is not False:
            raise ValueError("R18 runtime contract does not explicitly forbid Liger execution")
        if data_args.expected_arm_id not in qualification["scope"]["eligible_arm_ids"]:
            raise ValueError("hardware qualification permits C00 only")
        if data_args.sequence_length != qualification["training_unit"]["sequence_length"]:
            raise ValueError("hardware qualification sequence length drift")
    if training_args.h4_qualification_contract:
        h4, _ = load_h4_contract(Path(training_args.h4_qualification_contract))
        if qualification_sha256 != h4["parent"]["r18_machine_manifest_sha256"]:
            raise ValueError("H4 contract and R18 qualification manifest disagree")
        if training_args.h4_preregistration_closure_sha256 != H4_PREREGISTRATION_CLOSURE_SHA256:
            raise ValueError("H4 preregistration-closure identity drift")
        if training_args.h4_assay not in {"profiler", "timing"}:
            raise ValueError("--h4_assay must be profiler or timing when the H4 contract is active")
        if not training_args.require_no_dense_logits or not training_args.require_forward_loss_audit:
            raise ValueError("H4 requires both dense-logit rejection and every-forward loss auditing")
        if data_args.expected_arm_id != "C00" or training_args.world_size != 1:
            raise ValueError("H4 is a one-rank C00-only qualification")
        if training_args.gradient_accumulation_steps != 8 or training_args.per_device_train_batch_size != 1:
            raise ValueError("H4 batch geometry drift")
        if model_args.selected_loss_chunk_size not in h4["candidate_chunk_sizes_in_execution_order"]:
            raise ValueError("H4 candidate chunk size drift")
        schedule_contract = h4[f"{'four' if training_args.h4_assay == 'profiler' else 'thirteen'}_update_schedule"]
        expected_steps = 4 if training_args.h4_assay == "profiler" else 13
        if training_args.max_steps != expected_steps or expected_final != expected_steps:
            raise ValueError("H4 assay optimizer-step horizon drift")
        if data_args.expected_schedule_sha256 != schedule_contract["embedded_schedule_sha256"]:
            raise ValueError("H4 assay schedule identity drift")
        if h4_sha256_file(Path(data_args.pack_schedule_path)) != schedule_contract["file_sha256"]:
            raise ValueError("H4 assay schedule file digest drift")
        if training_args.hardware_profile is not (training_args.h4_assay == "profiler"):
            raise ValueError("H4 profiler flag/assay drift")
        if training_args.cuda_event_step_timing is not (training_args.h4_assay == "timing"):
            raise ValueError("H4 timing flag/assay drift")
        optimizer_name = getattr(training_args.optim, "value", training_args.optim)
        scheduler_name = getattr(training_args.lr_scheduler_type, "value", training_args.lr_scheduler_type)
        if optimizer_name != "adamw_torch_fused" or scheduler_name != "cosine":
            raise ValueError("H4 optimizer or scheduler implementation drift")
        exact_recipe = {
            "learning_rate": training_args.learning_rate,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "adam_epsilon": training_args.adam_epsilon,
            "weight_decay": training_args.weight_decay,
            "max_grad_norm": training_args.max_grad_norm,
            "warmup_ratio": training_args.warmup_ratio,
        }
        expected_recipe = {
            "learning_rate": 2e-5,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.03,
        }
        if exact_recipe != expected_recipe:
            raise ValueError(f"H4 optimizer hyperparameter drift: {exact_recipe}")
        if training_args.seed != 3407 or training_args.data_seed != 3407:
            raise ValueError("H4 seed drift")


def validate_frozen_data_contract(data_args: DataArguments, dataset: Qwen35NumpyPackedDataset) -> dict[str, Any]:
    manifest = dataset.manifest
    if manifest.get("suite_id") != EXPECTED_SUITE_ID:
        raise ValueError("NumPy dataset suite ID does not match the frozen core suite")
    if manifest.get("arm_id") != data_args.expected_arm_id:
        raise ValueError(f"NumPy dataset arm {manifest.get('arm_id')!r} != requested {data_args.expected_arm_id!r}")
    if manifest.get("renderer") != "qwen35_native_tools":
        raise ValueError("NumPy dataset was not produced by the native Qwen3.5 renderer")
    if manifest.get("enable_thinking") is not False:
        raise ValueError("NumPy dataset does not prove enable_thinking=false")
    if manifest.get("max_seq_length") != data_args.sequence_length:
        raise ValueError("NumPy renderer and trainer sequence lengths differ")
    if manifest.get("packing_semantics") != ("atomic_documents_best_fit_decreasing_no_cross_pack_or_part_splits"):
        raise ValueError("NumPy dataset packing-semantics declaration drift")
    if manifest.get("documents_index") != "documents.jsonl.gz":
        raise ValueError("NumPy dataset has no frozen stable document index")
    documents_index_sha256 = manifest.get("documents_index_sha256")
    if not isinstance(documents_index_sha256, str) or len(documents_index_sha256) != 64:
        raise ValueError("NumPy dataset has no valid document-index digest")
    tokenizer = manifest.get("tokenizer", {})
    if tokenizer.get("revision") != DEFAULT_MODEL_REVISION:
        raise ValueError("NumPy tokenizer revision drift")
    if tokenizer.get("chat_template_sha256") != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError("NumPy native chat-template digest drift")
    inputs = manifest.get("inputs", {})
    if inputs.get("core_operations_sha256") != EXPECTED_CORE_OPERATIONS_SHA256:
        raise ValueError("NumPy core-operation ledger digest drift")
    if inputs.get("frozen_design_manifest_sha256") != EXPECTED_FROZEN_DESIGN_SHA256:
        raise ValueError("NumPy frozen-design digest drift")
    amendment = manifest.get("renderer_amendment", {})
    if amendment.get("amendment_id") != EXPECTED_RENDERER_AMENDMENT_ID:
        raise ValueError("NumPy renderer-amendment ID drift")
    if amendment.get("manifest_sha256") != EXPECTED_RENDERER_AMENDMENT_SHA256:
        raise ValueError("NumPy renderer-amendment manifest digest drift")
    if set(amendment.get("observed_sample_uids", [])) != EXPECTED_AMENDED_SAMPLE_UIDS:
        raise ValueError("NumPy renderer-amendment UID set drift")
    accounting = dataset.accounting()
    if accounting["dropped_tokens"] != 0:
        raise ValueError("trainer packer dropped tokens despite the no-repeat policy")
    if accounting["packed_real_tokens"] != accounting["raw_tokens"]:
        raise ValueError("trainer packer did not retain every rendered token")
    if accounting["effective_trainable_tokens"] <= 0:
        raise ValueError("trainer dataset has no effective assistant supervision")
    return {
        "suite_id": manifest["suite_id"],
        "arm_id": manifest["arm_id"],
        "renderer": manifest["renderer"],
        "renderer_amendment_id": amendment["amendment_id"],
        "renderer_amendment_manifest_sha256": amendment["manifest_sha256"],
        "amended_sample_uids": sorted(amendment["observed_sample_uids"]),
        "tokenizer_revision": tokenizer["revision"],
        "chat_template_sha256": tokenizer["chat_template_sha256"],
        "documents_index": manifest["documents_index"],
        "documents_index_sha256": documents_index_sha256,
        "packing_accounting": accounting,
    }


def validate_saved_tokenizer(tokenizer: Qwen2TokenizerFast, dataset: Qwen35NumpyPackedDataset) -> dict[str, Any]:
    expected = dataset.manifest.get("tokenizer", {})
    chat_template = tokenizer.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("saved Qwen3.5 tokenizer has no chat template")
    observed = {
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "length": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "pad_token_id": tokenizer.pad_token_id,
    }
    for key in ("class", "vocab_size", "length", "chat_template_sha256"):
        if observed[key] != expected.get(key):
            raise ValueError(f"saved tokenizer {key} drift: {observed[key]!r} != {expected.get(key)!r}")
    if tokenizer.pad_token_id is None:
        raise ValueError("saved Qwen3.5 tokenizer has no pad token ID")
    return observed


def write_run_manifest(
    output_dir: Path,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: HFTrainingArguments,
    dataset: Qwen35NumpyPackedDataset,
    runtime_versions: dict[str, str],
    model: Qwen3_5ForCausalLM,
    frozen_data_validation: dict[str, Any],
    schedule_validation: dict[str, Any],
    formula: Qwen35FlopFormula,
    precision_validation: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    qualification_identity = None
    h4_identity = None
    if training_args.hardware_qualification_manifest:
        qualification, qualification_sha256 = load_qualification_manifest(
            Path(training_args.hardware_qualification_manifest)
        )
        qualification_identity = {
            "protocol_id": qualification["protocol_id"],
            "manifest_path": str(Path(training_args.hardware_qualification_manifest).resolve()),
            "manifest_sha256": qualification_sha256,
            "hardware_profile": bool(training_args.hardware_profile),
            "cuda_event_step_timing": bool(training_args.cuda_event_step_timing),
            "require_no_dense_logits": bool(training_args.require_no_dense_logits),
        }
    if training_args.h4_qualification_contract:
        h4, h4_sha256 = load_h4_contract(Path(training_args.h4_qualification_contract))
        h4_identity = {
            "assay": training_args.h4_assay,
            "candidate_chunk_size": model_args.selected_loss_chunk_size,
            "contract_path": str(Path(training_args.h4_qualification_contract).resolve()),
            "contract_sha256": h4_sha256,
            "preregistration_closure_sha256": training_args.h4_preregistration_closure_sha256,
            "protocol_id": h4["protocol_id"],
            "require_forward_loss_audit": bool(training_args.require_forward_loss_audit),
            "loaded_liger_modules_at_manifest": sorted(
                name for name in sys.modules if name == "liger_kernel" or name.startswith("liger_kernel.")
            ),
        }
    manifest = {
        "entry_point": "train_qwen35_sft.py",
        "model_name_or_path": model_args.model_name_or_path,
        "model_revision": model_args.model_revision,
        "model_class": type(model).__name__,
        "model_config_type": model.config.model_type,
        "model_text_vocab_size": model.config.vocab_size,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "vision_tower_loaded": False,
        "conditional_checkpoint_conversion": "strict_direct_to_Qwen3_5ForCausalLM",
        "text_conversion_ledger": "qwen35_text_conversion_ledger.json",
        "selective_output_projection": {
            "enabled": model_args.use_checkpointed_chunked_selected_loss,
            "implementation": CHUNKED_LOSS_IMPLEMENTATION_ID,
            "liger_status": "abandoned_after_r17",
            "chunk_size": model_args.selected_loss_chunk_size,
            "checkpoint": {
                "use_reentrant": False,
                "preserve_rng_state": True,
                "determinism_check": "default",
                "early_stop": True,
            },
            "row_semantics": "hidden[t] iff shifted_labels[t]=labels[t+1] is supervised",
        },
        "tokenizer_length": dataset.manifest["tokenizer"]["length"],
        "numpy_data_dir": str(Path(data_args.numpy_data_dir).resolve()),
        "numpy_contract_version": dataset.manifest["contract_version"],
        "numpy_manifest": dataset.manifest,
        "data_hash_verification": "identity_bearing_numpy_files_on_global_rank_0_before_model_load",
        "sequence_length": data_args.sequence_length,
        "num_packs": len(dataset),
        "dataset_accounting": dataset.accounting(),
        "frozen_data_validation": frozen_data_validation,
        "schedule_path": str(Path(data_args.pack_schedule_path).resolve()),
        "schedule_validation": schedule_validation,
        "drop_last": data_args.drop_last,
        "transformers_version": transformers.__version__,
        "expected_transformers_commit": PINNED_TRANSFORMERS_COMMIT,
        "verified_runtime_versions": runtime_versions,
        "precision_validation": precision_validation,
        "precision_policy": {
            "parameters": "FP32",
            "gradients": "FP32",
            "adamw_moments": "FP32",
            "forward_backward_autocast": "BF16",
        },
        "flop_formula": {**vars(formula), "formula_sha256": formula.formula_sha256},
        "world_size": training_args.world_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "training_arguments": training_args.to_dict(),
        "effective_tokens_per_optimizer_step": (
            data_args.sequence_length
            * training_args.world_size
            * training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size
        ),
        "hardware_qualification": qualification_identity,
        "h4_qualification": h4_identity,
    }
    write_json_atomic(output_dir / "qwen35_run_manifest.json", manifest)


def main() -> None:
    runtime_versions = verify_runtime()
    parser = HfArgumentParser((ModelArguments, DataArguments, Qwen35TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    validate_arguments(model_args, data_args, training_args)

    verify_data_files_on_this_rank = data_args.verify_data_hashes and training_args.process_index == 0
    base_dataset = Qwen35NumpyPackedDataset(
        data_args.numpy_data_dir,
        sequence_length=data_args.sequence_length,
        drop_last=data_args.drop_last,
        verify_hashes=verify_data_files_on_this_rank,
    )
    distributed_barrier_on_local_cuda_device()
    frozen_data_validation = validate_frozen_data_contract(data_args, base_dataset)
    if training_args.h4_qualification_contract:
        h4, _ = load_h4_contract(Path(training_args.h4_qualification_contract))
        numpy_manifest_path = Path(data_args.numpy_data_dir) / "manifest.json"
        if h4_sha256_file(numpy_manifest_path) != h4["data"]["numpy_manifest_sha256"]:
            raise ValueError("H4 C00 NumPy manifest digest drift")
        if frozen_data_validation["documents_index_sha256"] != h4["data"]["documents_index_sha256"]:
            raise ValueError("H4 C00 document-index digest drift")
    schedule = json.loads(Path(data_args.pack_schedule_path).read_text())
    global_packs_per_update = (
        training_args.world_size
        * training_args.gradient_accumulation_steps
        * training_args.per_device_train_batch_size
    )
    schedule_validation = validate_schedule_manifest(
        schedule,
        base_dataset,
        expected_seed=training_args.seed,
        expected_global_packs_per_update=global_packs_per_update,
    )
    if schedule_validation["schedule_sha256"] != data_args.expected_schedule_sha256:
        raise ValueError("validated schedule digest does not equal --expected_schedule_sha256")
    if schedule_validation["optimizer_updates"] != training_args.max_steps:
        raise ValueError(
            f"schedule contains {schedule_validation['optimizer_updates']} updates, but max_steps is "
            f"{training_args.max_steps}"
        )
    dataset = ScheduledQwen35Dataset(base_dataset, schedule)
    tokenizer_path = Path(data_args.numpy_data_dir) / base_dataset.manifest["tokenizer"]["directory"]
    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)
    frozen_data_validation["runtime_tokenizer_validation"] = validate_saved_tokenizer(tokenizer, base_dataset)

    full_config = Qwen3_5Config.from_pretrained(
        model_args.model_name_or_path, revision=model_args.model_revision, cache_dir=model_args.cache_dir
    )
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=full_config.text_config,
        revision=model_args.model_revision,
        cache_dir=model_args.cache_dir,
        dtype=torch.float32,
        attn_implementation="flash_attention_2",
        output_loading_info=True,
    )
    validate_text_loading_info(loading_info)
    if model.config.model_type != "qwen3_5_text":
        raise ValueError(f"expected qwen3_5_text model_type, found {model.config.model_type!r}")
    if len(tokenizer) > model.config.vocab_size:
        raise ValueError(
            f"tokenizer contains IDs outside the model output vocabulary: {len(tokenizer)} > {model.config.vocab_size}"
        )
    model.config.use_cache = False
    precision_validation = validate_fp32_trainable_parameters(model)
    install_qwen35_checkpointed_chunked_loss(model, chunk_size=model_args.selected_loss_chunk_size)
    if model.forward.__module__ != "open_instruct.qwen35_chunked_loss":
        raise RuntimeError("R18 checkpointed chunked loss did not replace Qwen3.5 CausalLM.forward")
    verify_packed_kernel_support(model)
    formula = Qwen35FlopFormula.from_config(model.config)

    output_dir = Path(training_args.output_dir)
    if training_args.should_save:
        conversion_ledger = build_text_conversion_ledger(
            model,
            source_model=model_args.model_name_or_path,
            source_revision=model_args.model_revision,
            hash_tensors=model_args.hash_text_conversion_tensors,
        )
        write_json_atomic(output_dir / "qwen35_text_conversion_ledger.json", conversion_ledger)

    collator = Qwen35PackedCollator(
        pad_token_id=tokenizer.pad_token_id,
        sequence_length=data_args.sequence_length,
        pad_to_sequence_length=not data_args.drop_last,
    )
    if training_args.should_save:
        write_run_manifest(
            output_dir,
            model_args,
            data_args,
            training_args,
            base_dataset,
            runtime_versions,
            model,
            frozen_data_validation,
            schedule_validation,
            formula,
            precision_validation,
        )
    distributed_barrier_on_local_cuda_device()
    expected_final_global_step = training_args.expected_final_global_step
    if expected_final_global_step is None:
        expected_final_global_step = training_args.stop_after_steps or training_args.max_steps
    exact_metrics = Qwen35ExactMetricsCallback(
        output_dir=output_dir,
        sequence_length=data_args.sequence_length,
        schedule_sha256=schedule_validation["schedule_sha256"],
        formula=formula,
        expected_initial_global_step=training_args.expected_initial_global_step,
        expected_final_global_step=expected_final_global_step,
        sync_interval=training_args.exact_metrics_sync_interval,
    )
    trainer = Qwen35ExactTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        exact_metrics=exact_metrics,
        callbacks=[exact_metrics],
    )
    trainer.add_callback(
        Qwen35UpdateProbeCallback(output_dir, training_args.expected_initial_global_step, expected_final_global_step)
    )
    if training_args.hardware_profile:
        if training_args.expected_initial_global_step != 0:
            raise ValueError("hardware profiler is only supported for an initial bounded trajectory")
        trainer.add_callback(
            Qwen35HardwareProfilerCallback(
                qualification_manifest_path=Path(training_args.hardware_qualification_manifest),
                h4_contract_path=(
                    Path(training_args.h4_qualification_contract)
                    if training_args.h4_qualification_contract
                    else None
                ),
                output_dir=output_dir,
                expected_steps=expected_final_global_step,
                candidate_chunk_size=model_args.selected_loss_chunk_size,
            )
        )
    if training_args.cuda_event_step_timing:
        trainer.add_callback(
            Qwen35CudaEventTimerCallback(
                qualification_manifest_path=Path(training_args.hardware_qualification_manifest),
                h4_contract_path=(
                    Path(training_args.h4_qualification_contract)
                    if training_args.h4_qualification_contract
                    else None
                ),
                output_dir=output_dir,
                expected_steps=expected_final_global_step,
                candidate_chunk_size=model_args.selected_loss_chunk_size,
            )
        )
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    distributed_barrier_on_local_cuda_device()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Do not put a barrier in this exceptional path: an asymmetric failure
        # must not be converted into a distributed deadlock.
        destroy_initialized_process_group()
