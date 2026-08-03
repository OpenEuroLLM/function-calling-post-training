"""Qwen3.5-specific counters and analytic MFU reporting."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 1
NOMINAL_A100_DENSE_BF16_FLOPS_PER_SECOND = 312_000_000_000_000


@dataclass(frozen=True)
class Qwen35FlopFormula:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_gdn_layers: int
    num_full_attention_layers: int
    full_attention_heads: int
    full_attention_kv_heads: int
    full_attention_head_dim: int
    gdn_heads: int
    gdn_key_head_dim: int
    gdn_value_head_dim: int
    vocabulary_size: int
    decoder_linear_training_flops_per_fixed_token: int
    gdn_training_flops_per_fixed_token: int
    nominal_peak_flops_per_second_per_gpu: int = NOMINAL_A100_DENSE_BF16_FLOPS_PER_SECOND
    formula_version: str = "qwen35-hybrid-causal-selected-output-v2"

    @property
    def formula_sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_config(cls, config: Any) -> Qwen35FlopFormula:
        layer_types = list(config.layer_types)
        num_gdn_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
        num_full_attention_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
        if num_gdn_layers + num_full_attention_layers != config.num_hidden_layers:
            raise ValueError("Qwen3.5 config contains an unsupported layer type")
        if config.linear_num_key_heads != config.linear_num_value_heads:
            raise ValueError("the Qwen3.5 FLOP formula requires equal GDN key/value head counts")
        gdn_key_width = config.linear_num_key_heads * config.linear_key_head_dim
        gdn_value_width = config.linear_num_value_heads * config.linear_value_head_dim
        mlp_weights = config.num_hidden_layers * 3 * config.hidden_size * config.intermediate_size
        full_weights_per_layer = (
            config.hidden_size * (2 * config.num_attention_heads * config.head_dim)
            + 2 * config.hidden_size * (config.num_key_value_heads * config.head_dim)
            + config.num_attention_heads * config.head_dim * config.hidden_size
        )
        gdn_weights_per_layer = (
            config.hidden_size * (2 * gdn_key_width + gdn_value_width)
            + config.hidden_size * gdn_value_width
            + 2 * config.hidden_size * config.linear_num_key_heads
            + gdn_value_width * config.hidden_size
        )
        decoder_linear_weights = (
            mlp_weights + num_full_attention_layers * full_weights_per_layer + num_gdn_layers * gdn_weights_per_layer
        )
        gdn_state_elements = config.linear_num_key_heads * config.linear_key_head_dim * config.linear_value_head_dim
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            num_gdn_layers=num_gdn_layers,
            num_full_attention_layers=num_full_attention_layers,
            full_attention_heads=config.num_attention_heads,
            full_attention_kv_heads=config.num_key_value_heads,
            full_attention_head_dim=config.head_dim,
            gdn_heads=config.linear_num_key_heads,
            gdn_key_head_dim=config.linear_key_head_dim,
            gdn_value_head_dim=config.linear_value_head_dim,
            vocabulary_size=config.vocab_size,
            decoder_linear_training_flops_per_fixed_token=6 * decoder_linear_weights,
            gdn_training_flops_per_fixed_token=3 * num_gdn_layers * 7 * gdn_state_elements,
        )

    def window_flops(
        self, *, fixed_tokens: int, assistant_targets: int, attention_length_squared: int
    ) -> dict[str, int]:
        if min(fixed_tokens, assistant_targets, attention_length_squared) < 0:
            raise ValueError("FLOP counter inputs must be nonnegative")
        decoder = self.decoder_linear_training_flops_per_fixed_token * fixed_tokens
        gdn = self.gdn_training_flops_per_fixed_token * fixed_tokens
        causal_attention_pairs = self.isolated_causal_attention_pairs(
            fixed_tokens=fixed_tokens, attention_length_squared=attention_length_squared
        )
        attention = (
            12
            * self.num_full_attention_layers
            * self.full_attention_heads
            * self.full_attention_head_dim
            * causal_attention_pairs
        )
        selected_output = 6 * self.hidden_size * self.vocabulary_size * assistant_targets
        return {
            "decoder_linear_and_mlp": decoder,
            "gdn_recurrence_approximation": gdn,
            "document_isolated_causal_full_attention": attention,
            "selected_output_projection": selected_output,
            "total": decoder + gdn + attention + selected_output,
        }

    @staticmethod
    def isolated_causal_attention_pairs(*, fixed_tokens: int, attention_length_squared: int) -> int:
        if min(fixed_tokens, attention_length_squared) < 0:
            raise ValueError("attention-pair counter inputs must be nonnegative")
        causal_pair_numerator = attention_length_squared + fixed_tokens
        if causal_pair_numerator % 2:
            raise ValueError("isolated causal-attention pair count is not integral")
        return causal_pair_numerator // 2


@dataclass
class Qwen35WindowCounts:
    fixed_tokens: int = 0
    real_tokens: int = 0
    assistant_targets: int = 0
    padding_tokens: int = 0
    attention_length_squared: int = 0
    documents: int = 0
    packs: int = 0
    synthetic_packs: int = 0

    def add_pack(self, metadata: dict[str, Any], sequence_length: int) -> None:
        self.fixed_tokens += sequence_length
        self.real_tokens += int(metadata["real_tokens"])
        self.assistant_targets += int(metadata["assistant_targets"])
        self.padding_tokens += int(metadata["padding_tokens"])
        self.attention_length_squared += int(metadata["attention_length_squared"])
        self.documents += int(metadata["document_count"])
        self.packs += 1
        self.synthetic_packs += int(bool(metadata["synthetic"]))

    def add(self, other: Qwen35WindowCounts) -> None:
        for key in asdict(self):
            setattr(self, key, getattr(self, key) + getattr(other, key))


def build_reporting_record(
    *,
    formula: Qwen35FlopFormula,
    step: int,
    world_size: int,
    elapsed_seconds: float,
    counts: Qwen35WindowCounts,
    schedule_sha256: str,
    pack_uids: list[str],
    schedule_indices: list[int],
    learning_rate: float | None,
    normalized_loss: float | None,
    global_target_divisor: int | None,
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
    synchronized: bool,
    host_timing_seconds: dict[str, float] | None = None,
    optimizer_updates: int = 1,
    applied_learning_rates: list[float] | None = None,
) -> dict[str, Any]:
    if step <= 0:
        raise ValueError("reporting step must be positive")
    if elapsed_seconds <= 0 or not math.isfinite(elapsed_seconds):
        raise ValueError("reporting elapsed time must be finite and positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if optimizer_updates <= 0:
        raise ValueError("optimizer_updates must be positive")
    if any(value < 0 for value in asdict(counts).values()):
        raise ValueError("reporting counts must be nonnegative")
    if counts.real_tokens + counts.padding_tokens != counts.fixed_tokens:
        raise ValueError("real-token and padding-token counts do not sum to fixed tokens")
    if counts.assistant_targets > counts.real_tokens:
        raise ValueError("assistant targets cannot exceed real tokens")
    if len(pack_uids) != counts.packs or len(schedule_indices) != counts.packs:
        raise ValueError("pack identity lists do not match the pack count")
    if len(set(pack_uids)) != len(pack_uids) or len(set(schedule_indices)) != len(schedule_indices):
        raise ValueError("a reporting window contains a repeated pack or schedule index")
    if global_target_divisor != counts.assistant_targets:
        raise ValueError("global loss divisor does not equal the window assistant-target count")
    if normalized_loss is not None and (not math.isfinite(normalized_loss) or normalized_loss < 0):
        raise ValueError("normalized loss must be finite and nonnegative")
    if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate < 0):
        raise ValueError("learning rate must be finite and nonnegative")
    if applied_learning_rates is None:
        applied_learning_rates = [] if learning_rate is None else [learning_rate] * optimizer_updates
    if len(applied_learning_rates) != optimizer_updates:
        raise ValueError("applied learning-rate count must equal optimizer updates")
    if any(not math.isfinite(value) or value < 0 for value in applied_learning_rates):
        raise ValueError("applied learning rates must be finite and nonnegative")
    if learning_rate is not None and applied_learning_rates[-1] != learning_rate:
        raise ValueError("learning_rate must equal the final applied learning rate")
    for name, value in (("peak_allocated_bytes", peak_allocated_bytes), ("peak_reserved_bytes", peak_reserved_bytes)):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if (
        peak_allocated_bytes is not None
        and peak_reserved_bytes is not None
        and peak_allocated_bytes > peak_reserved_bytes
    ):
        raise ValueError("allocated CUDA bytes cannot exceed reserved CUDA bytes")
    host_timing_seconds = host_timing_seconds or {}
    if any(value < 0 or not math.isfinite(value) for value in host_timing_seconds.values()):
        raise ValueError("host timing values must be finite and nonnegative")
    flops = formula.window_flops(
        fixed_tokens=counts.fixed_tokens,
        assistant_targets=counts.assistant_targets,
        attention_length_squared=counts.attention_length_squared,
    )
    peak = formula.nominal_peak_flops_per_second_per_gpu
    return {
        "artifact": "qwen35_training_step_metrics",
        "schema_version": REPORT_SCHEMA_VERSION,
        "step": step,
        "window_start_step": step - optimizer_updates + 1,
        "optimizer_updates": optimizer_updates,
        "schedule_sha256": schedule_sha256,
        "schedule_indices": schedule_indices,
        "pack_uids": pack_uids,
        "world_size": world_size,
        "synchronized_timing": synchronized,
        "elapsed_seconds": elapsed_seconds,
        "timing": {
            "synchronized_window_wall_seconds": elapsed_seconds,
            "host_enqueue_durations_seconds": host_timing_seconds,
            "host_timing_caveat": (
                "Host enqueue durations are diagnostic and can overlap asynchronous CUDA work; only the "
                "synchronized window wall time is an end-to-end duration. Device-region timing requires CUDA events."
            ),
        },
        "counts": asdict(counts),
        "rates": {
            "fixed_tokens_per_second_global": counts.fixed_tokens / elapsed_seconds,
            "fixed_tokens_per_second_per_gpu": counts.fixed_tokens / (world_size * elapsed_seconds),
            "real_tokens_per_second_global": counts.real_tokens / elapsed_seconds,
            "assistant_targets_per_second_global": counts.assistant_targets / elapsed_seconds,
            "optimizer_steps_per_second": optimizer_updates / elapsed_seconds,
        },
        "loss": {"normalized_loss": normalized_loss, "global_assistant_target_divisor": global_target_divisor},
        "optimizer": {
            "learning_rate": learning_rate,
            "applied_learning_rates": applied_learning_rates,
            "first_applied_learning_rate": applied_learning_rates[0] if applied_learning_rates else None,
            "last_applied_learning_rate": applied_learning_rates[-1] if applied_learning_rates else None,
        },
        "memory": {"peak_allocated_bytes": peak_allocated_bytes, "peak_reserved_bytes": peak_reserved_bytes},
        "analytic_flops": {
            "formula_version": formula.formula_version,
            "formula_sha256": formula.formula_sha256,
            "components": flops,
            "isolated_causal_attention_pairs": formula.isolated_causal_attention_pairs(
                fixed_tokens=counts.fixed_tokens, attention_length_squared=counts.attention_length_squared
            ),
            "nominal_peak_flops_per_second_per_gpu": peak,
            "analytic_model_mfu": flops["total"] / (world_size * peak * elapsed_seconds),
            "caveat": (
                "Useful-model FLOP convention; excludes optimizer, minor elementwise work, "
                "and implementation-specific recomputation. Nominal A100 peak is not an in-situ measurement."
            ),
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize_reporting_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty Qwen3.5 metrics stream")
    expected_artifact = "qwen35_training_step_metrics"
    schedule_sha256 = records[0]["schedule_sha256"]
    formula_version = records[0]["analytic_flops"]["formula_version"]
    formula_sha256 = records[0]["analytic_flops"]["formula_sha256"]
    world_size = records[0]["world_size"]
    previous_step = None
    optimizer_steps = 0
    seen_schedule_indices: set[int] = set()
    total_flops = 0
    for record in records:
        if record.get("artifact") != expected_artifact or record.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise ValueError("metrics stream contains an unsupported record")
        if record["schedule_sha256"] != schedule_sha256:
            raise ValueError("metrics stream mixes schedules")
        if record["analytic_flops"]["formula_version"] != formula_version:
            raise ValueError("metrics stream mixes FLOP formula versions")
        if record["analytic_flops"]["formula_sha256"] != formula_sha256:
            raise ValueError("metrics stream mixes FLOP formula parameters")
        if record["world_size"] != world_size:
            raise ValueError("metrics stream mixes world sizes")
        step = int(record["step"])
        window_updates = int(record.get("optimizer_updates", 0))
        if window_updates <= 0 or record.get("window_start_step") != step - window_updates + 1:
            raise ValueError("metrics stream contains invalid optimizer-window accounting")
        if previous_step is not None and record["window_start_step"] != previous_step + 1:
            raise ValueError("metrics steps are not contiguous")
        previous_step = step
        optimizer_steps += window_updates
        overlap = seen_schedule_indices.intersection(record["schedule_indices"])
        if overlap:
            raise ValueError(f"metrics stream repeats schedule indices: {sorted(overlap)[:10]}")
        seen_schedule_indices.update(record["schedule_indices"])
        total_flops += int(record["analytic_flops"]["components"]["total"])

    elapsed = sum(float(record["elapsed_seconds"]) for record in records)
    counts = Qwen35WindowCounts()
    for record in records:
        counts.add(Qwen35WindowCounts(**record["counts"]))
    return {
        "artifact": "qwen35_training_metrics_summary",
        "schema_version": REPORT_SCHEMA_VERSION,
        "reporting_windows": len(records),
        "optimizer_steps": optimizer_steps,
        "first_step": records[0]["window_start_step"],
        "last_step": records[-1]["step"],
        "elapsed_seconds": elapsed,
        "counts": asdict(counts),
        "aggregate_rates": {
            "fixed_tokens_per_second_global": counts.fixed_tokens / elapsed,
            "real_tokens_per_second_global": counts.real_tokens / elapsed,
            "assistant_targets_per_second_global": counts.assistant_targets / elapsed,
        },
        "analytic_flops": {
            "formula_version": formula_version,
            "formula_sha256": formula_sha256,
            "total": total_flops,
            "analytic_model_mfu": total_flops
            / (world_size * int(records[0]["analytic_flops"]["nominal_peak_flops_per_second_per_gpu"]) * elapsed),
        },
        "schedule_sha256": schedule_sha256,
        "world_size": world_size,
    }
