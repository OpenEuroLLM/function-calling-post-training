"""CUDA profiler and memory evidence for the Qwen3.5 qualification path."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from open_instruct.qwen35_qualification import validate_memory_headroom
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_h4 import (
    H4_ALLOCATOR_HISTORY_ENTRY_CAP,
    load_h4_contract,
    sha256_file,
)
from open_instruct.qwen35_training import write_json_atomic


class Qwen35CudaEventTimerCallback(TrainerCallback):
    """Record per-rank per-update CUDA event durations without per-step synchronization."""

    def __init__(
        self,
        *,
        qualification_manifest_path: Path,
        h4_contract_path: Path | None,
        output_dir: Path,
        expected_steps: int,
        candidate_chunk_size: int | None,
    ) -> None:
        self.manifest, self.manifest_sha256 = load_qualification_manifest(qualification_manifest_path)
        self.h4 = None
        self.h4_sha256 = None
        if h4_contract_path is not None:
            self.h4, self.h4_sha256 = load_h4_contract(h4_contract_path)
            if self.manifest_sha256 != self.h4["parent"]["r18_machine_manifest_sha256"]:
                raise ValueError("CUDA-event timer R18/H4 manifest identity drift")
            if expected_steps != int(self.h4["timing_assay"]["maximum_updates"]):
                raise ValueError("CUDA-event timer must use the frozen thirteen-update H4 timing assay")
            if candidate_chunk_size not in self.h4["candidate_chunk_sizes_in_execution_order"]:
                raise ValueError("CUDA-event timer received an unknown H4 candidate")
        self.output_dir = output_dir
        self.expected_steps = expected_steps
        self.candidate_chunk_size = None if candidate_chunk_size is None else int(candidate_chunk_size)
        self.starts: dict[int, torch.cuda.Event] = {}
        self.ends: dict[int, torch.cuda.Event] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-event timer requires CUDA")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        step = int(state.global_step) + 1
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.starts[step] = event
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.ends[step] = event
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if int(state.global_step) != self.expected_steps:
            raise RuntimeError(f"CUDA-event timer ended at {state.global_step}, expected {self.expected_steps}")
        if self.starts.keys() != self.ends.keys() or len(self.starts) != self.expected_steps:
            raise RuntimeError("CUDA-event timer did not observe every optimizer step")
        torch.cuda.synchronize()
        durations = {str(step): float(self.starts[step].elapsed_time(self.ends[step])) for step in sorted(self.starts)}
        if any(not math.isfinite(value) or value <= 0 for value in durations.values()):
            raise RuntimeError("CUDA-event timer observed a nonpositive or nonfinite duration")
        rank = int(args.process_index)
        report = {
            "artifact": "qwen35_per_rank_cuda_event_step_timing",
            "schema_version": 1,
            "status": "passed",
            "qualification_protocol_id": self.manifest["protocol_id"],
            "qualification_manifest_sha256": self.manifest_sha256,
            "h4_protocol_id": None if self.h4 is None else self.h4["protocol_id"],
            "h4_contract_sha256": self.h4_sha256,
            "assay": "timing",
            "candidate_chunk_size": self.candidate_chunk_size,
            "rank": rank,
            "world_size": int(args.world_size),
            "completed_optimizer_steps": int(state.global_step),
            "cuda_event_step_milliseconds": durations,
            "timing_scope": "rank-local default-stream events around Trainer optimizer step; synchronized once at train end",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.output_dir / f"qwen35_cuda_step_times_rank{rank:02d}.json", report)
        return control


class Qwen35HardwareProfilerCallback(TrainerCallback):
    """Capture a bounded CUDA trace, allocator snapshot, event timing, and HBM gate."""

    HISTORY_ENTRY_CAP = H4_ALLOCATOR_HISTORY_ENTRY_CAP

    def __init__(
        self,
        *,
        qualification_manifest_path: Path,
        h4_contract_path: Path | None,
        output_dir: Path,
        expected_steps: int,
        candidate_chunk_size: int | None,
    ) -> None:
        self.manifest, self.manifest_sha256 = load_qualification_manifest(qualification_manifest_path)
        self.h4 = None
        self.h4_sha256 = None
        if h4_contract_path is None:
            if expected_steps < 2:
                raise ValueError("hardware profiler requires at least one warmup and one active step")
        else:
            self.h4, self.h4_sha256 = load_h4_contract(h4_contract_path)
            if self.manifest_sha256 != self.h4["parent"]["r18_machine_manifest_sha256"]:
                raise ValueError("hardware profiler R18/H4 manifest identity drift")
            if expected_steps != int(self.h4["profiler_assay"]["maximum_updates"]):
                raise ValueError("hardware profiler must use the frozen four-update H4 profiler assay")
            if candidate_chunk_size not in self.h4["candidate_chunk_sizes_in_execution_order"]:
                raise ValueError("hardware profiler received an unknown H4 candidate")
        self.output_dir = output_dir
        self.expected_steps = expected_steps
        self.candidate_chunk_size = None if candidate_chunk_size is None else int(candidate_chunk_size)
        self.profiler: Any | None = None
        self.step_start_events: dict[int, torch.cuda.Event] = {}
        self.step_event_milliseconds: dict[int, float] = {}
        self.step_peak_allocated_bytes: dict[int, int] = {}
        self.step_peak_reserved_bytes: dict[int, int] = {}
        self.trace_path = output_dir / "qwen35_cuda_profiler_trace.json"
        self.memory_snapshot_path = output_dir / "qwen35_cuda_memory_snapshot.pickle"
        self.initial_free_device_bytes: int | None = None
        self.total_device_bytes: int | None = None

    def _trace_ready(self, profiler: Any) -> None:
        if self.trace_path.exists():
            raise FileExistsError(self.trace_path)
        profiler.export_chrome_trace(str(self.trace_path))

    def on_train_begin(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("hardware-profiler callback requires CUDA")
        if not state.is_world_process_zero:
            return control
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.trace_path.exists() or self.memory_snapshot_path.exists():
            raise FileExistsError("hardware-profiler output already exists")
        torch.cuda.synchronize()
        initial_free, total = torch.cuda.mem_get_info()
        self.initial_free_device_bytes = int(initial_free)
        self.total_device_bytes = int(total)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory.reset_accumulated_memory_stats()
        torch.cuda.memory._record_memory_history(  # noqa: SLF001 - pinned qualification-only allocator API
            enabled="all", context="all", stacks="python", max_entries=self.HISTORY_ENTRY_CAP, clear_history=True
        )
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=self.expected_steps - 1, repeat=1),
            on_trace_ready=self._trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        self.profiler.__enter__()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.step_start_events[int(state.global_step) + 1] = event
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        step = int(state.global_step)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        start = self.step_start_events.pop(step)
        self.step_event_milliseconds[step] = float(start.elapsed_time(end))
        self.step_peak_allocated_bytes[step] = int(torch.cuda.max_memory_allocated())
        self.step_peak_reserved_bytes[step] = int(torch.cuda.max_memory_reserved())
        if self.profiler is None:
            raise RuntimeError("hardware profiler was not initialized")
        self.profiler.step()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        if int(state.global_step) != self.expected_steps:
            raise RuntimeError(f"hardware profile ended at step {state.global_step}, expected {self.expected_steps}")
        if self.profiler is None:
            raise RuntimeError("hardware profiler was not initialized")
        self.profiler.__exit__(None, None, None)
        torch.cuda.synchronize()
        try:
            torch.cuda.memory._dump_snapshot(str(self.memory_snapshot_path))  # noqa: SLF001
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)  # noqa: SLF001
        if not self.trace_path.is_file() or self.trace_path.stat().st_size <= 0:
            raise RuntimeError("hardware profiler did not export a nonempty Chrome trace")
        if not self.memory_snapshot_path.is_file() or self.memory_snapshot_path.stat().st_size <= 0:
            raise RuntimeError("hardware profiler did not export a nonempty allocator snapshot")
        expected_step_set = set(range(1, self.expected_steps + 1))
        if set(self.step_peak_allocated_bytes) != expected_step_set or set(self.step_peak_reserved_bytes) != expected_step_set:
            raise RuntimeError("hardware profiler did not capture CUDA memory peaks for every optimizer step")
        peak_allocated = max(self.step_peak_allocated_bytes.values())
        peak_reserved = max(self.step_peak_reserved_bytes.values())
        free_device_bytes, total_device_bytes = torch.cuda.mem_get_info()
        free_device_bytes = int(free_device_bytes)
        total_device_bytes = int(total_device_bytes)
        if self.total_device_bytes != total_device_bytes:
            raise RuntimeError("CUDA device total memory changed during the H4 profiler assay")
        memory = validate_memory_headroom(
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            total_device_bytes=total_device_bytes,
            acceptance=(self.h4 or self.manifest)["memory_acceptance"],
        )
        memory_stats = torch.cuda.memory_stats()
        allocator_retries = int(memory_stats.get("num_alloc_retries", 0))
        if allocator_retries > int((self.h4 or self.manifest)["memory_acceptance"]["maximum_allocator_retries"]):
            raise AssertionError("CUDA allocator retries exceed the frozen qualification threshold")
        report: dict[str, Any] = {
            "artifact": "qwen35_cuda_hardware_profile",
            "schema_version": 1,
            "status": "captured_pending_kernel_audit",
            "qualification_protocol_id": self.manifest["protocol_id"],
            "qualification_manifest_sha256": self.manifest_sha256,
            "h4_protocol_id": None if self.h4 is None else self.h4["protocol_id"],
            "h4_contract_sha256": self.h4_sha256,
            "assay": "profiler",
            "candidate_chunk_size": self.candidate_chunk_size,
            "pid": os.getpid(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_device_capability": list(torch.cuda.get_device_capability()),
            "completed_optimizer_steps": int(state.global_step),
            "profiler_schedule": {
                "wait": 0,
                "warmup": 1,
                "active": self.expected_steps - 1,
                "repeat": 1,
                "skip_first": 0,
            },
            "warmup_optimizer_steps": 1,
            "measured_optimizer_steps": self.expected_steps - 1,
            "cuda_event_step_milliseconds": {
                str(step): milliseconds for step, milliseconds in sorted(self.step_event_milliseconds.items())
            },
            "memory": memory,
            "per_step_memory": {
                "peak_allocated_bytes": {
                    str(step): value for step, value in sorted(self.step_peak_allocated_bytes.items())
                },
                "peak_reserved_bytes": {
                    str(step): value for step, value in sorted(self.step_peak_reserved_bytes.items())
                },
                "aggregation": "maximum_across_all_four_steps_after_exact_metrics_window_resets",
            },
            "allocator": {"num_alloc_retries": allocator_retries, "num_ooms": int(memory_stats.get("num_ooms", 0))},
            "device_memory_observations": {
                "initial_free_bytes_after_model_load": self.initial_free_device_bytes,
                "final_free_bytes": free_device_bytes,
                "total_bytes": total_device_bytes,
            },
            "allocator_history": {
                "enabled": "all",
                "context": "all",
                "stacks": "python",
                "maximum_entries": self.HISTORY_ENTRY_CAP,
                "clear_history": True,
            },
            "trace_path": str(self.trace_path),
            "trace_bytes": self.trace_path.stat().st_size,
            "trace_sha256": sha256_file(self.trace_path),
            "memory_snapshot_path": str(self.memory_snapshot_path),
            "memory_snapshot_bytes": self.memory_snapshot_path.stat().st_size,
            "memory_snapshot_sha256": sha256_file(self.memory_snapshot_path),
        }
        if report["allocator"]["num_ooms"] != 0:
            raise AssertionError("CUDA allocator recorded an out-of-memory event")
        write_json_atomic(self.output_dir / "qwen35_cuda_hardware_profile.json", report)
        return control
