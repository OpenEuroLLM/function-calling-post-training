#!/usr/bin/env python3
"""Validate one completed G2 output before its Slurm job may be accepted."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from open_instruct.qwen35_qualification import validate_h1_reference_report

PINNED_LIGER_COMMIT = "72a4ed47a5c593b58045a0af14d3f774a037bd92"
EXPECTED_FLOP_FORMULA_VERSION = "qwen35-hybrid-causal-selected-output-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-initial-step", type=int, required=True)
    parser.add_argument("--expected-final-step", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-gradient-accumulation", type=int, required=True)
    parser.add_argument("--expected-schedule-sha256", required=True)
    parser.add_argument("--expected-qualification-manifest-sha256", required=True)
    parser.add_argument("--require-reference-parity", action="store_true")
    parser.add_argument("--require-generation-parser", action="store_true")
    parser.add_argument("--require-resume-parity", action="store_true")
    parser.add_argument("--require-conversion-parity", action="store_true")
    parser.add_argument("--require-selective-loss", action="store_true")
    parser.add_argument("--require-ddp-normalization", action="store_true")
    parser.add_argument("--require-schedule-sharding", action="store_true")
    parser.add_argument("--require-hardware-profile", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite_json(value: Any, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value in {context}")
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_json(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_json(child, context=f"{context}[{index}]")


def _close(observed: float, expected: float, *, context: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{context} drift: observed={observed}, expected={expected}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    if args.expected_initial_step < 0 or args.expected_final_step <= args.expected_initial_step:
        raise ValueError("invalid expected G2 step interval")
    expected_checkpoint = root / f"checkpoint-{args.expected_final_step}"
    required = {
        "run_manifest": root / "qwen35_run_manifest.json",
        "update_probe": root / "qwen35_parameter_update_probe.json",
        "conversion_ledger": root / "qwen35_text_conversion_ledger.json",
        "exact_metrics_summary": root / "qwen35_exact_metrics_summary.json",
        "exact_metrics": root / "qwen35_exact_metrics.jsonl",
        "trainer_state": root / "trainer_state.json",
        "train_results": root / "train_results.json",
        "checkpoint_trainer_state": expected_checkpoint / "trainer_state.json",
        "checkpoint_model": expected_checkpoint / "model.safetensors",
        "checkpoint_optimizer": expected_checkpoint / "optimizer.pt",
        "checkpoint_scheduler": expected_checkpoint / "scheduler.pt",
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing G2 {label}: {path}")
    expected_rng_files = (
        [expected_checkpoint / "rng_state.pth"]
        if args.expected_world_size == 1
        else [expected_checkpoint / f"rng_state_{rank}.pth" for rank in range(args.expected_world_size)]
    )
    for path in expected_rng_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing distributed RNG checkpoint: {path}")

    run_manifest = read_json(required["run_manifest"])
    update_probe = read_json(required["update_probe"])
    conversion_ledger = read_json(required["conversion_ledger"])
    exact_metrics_summary = read_json(required["exact_metrics_summary"])
    exact_metrics = [json.loads(line) for line in required["exact_metrics"].read_text().splitlines() if line.strip()]
    trainer_state = read_json(required["trainer_state"])
    checkpoint_state = read_json(required["checkpoint_trainer_state"])
    train_results = read_json(required["train_results"])
    for name, value in (
        ("run_manifest", run_manifest),
        ("update_probe", update_probe),
        ("conversion_ledger", conversion_ledger),
        ("exact_metrics_summary", exact_metrics_summary),
        ("exact_metrics", exact_metrics),
        ("trainer_state", trainer_state),
        ("checkpoint_state", checkpoint_state),
        ("train_results", train_results),
    ):
        require_finite_json(value, context=name)

    qualification_identity = run_manifest.get("hardware_qualification")
    if not isinstance(qualification_identity, dict):
        raise ValueError("G2 run manifest lacks hardware-qualification identity")
    if qualification_identity.get("protocol_id") != "qwen35-hardware-qualification-r15":
        raise ValueError("G2 run manifest hardware-qualification protocol drift")
    if qualification_identity.get("manifest_sha256") != args.expected_qualification_manifest_sha256:
        raise ValueError("G2 run manifest hardware-qualification manifest drift")
    if qualification_identity.get("require_no_dense_logits") is not True:
        raise ValueError("G2 run did not require full-logit avoidance")

    if run_manifest.get("frozen_data_validation", {}).get("arm_id") != "C00":
        raise ValueError("G2 did not consume frozen C00")
    if run_manifest.get("sequence_length") != 32768 or run_manifest.get("drop_last") is not False:
        raise ValueError("G2 sequence length or no-repeat packing policy drift")
    if run_manifest.get("data_hash_verification") != "identity_bearing_numpy_files_on_global_rank_0_before_model_load":
        raise ValueError("G2 full-file hash verification policy drift")
    tokenizer_validation = run_manifest.get("frozen_data_validation", {}).get("runtime_tokenizer_validation", {})
    if (
        tokenizer_validation.get("class") != "Qwen2TokenizerFast"
        or tokenizer_validation.get("chat_template_sha256") is None
        or tokenizer_validation.get("pad_token_id") is None
    ):
        raise ValueError("G2 saved-tokenizer validation evidence drift")
    if run_manifest.get("world_size") != args.expected_world_size:
        raise ValueError("G2 world size drift")
    if run_manifest.get("gradient_accumulation_steps") != args.expected_gradient_accumulation:
        raise ValueError("G2 gradient accumulation drift")
    if run_manifest.get("model_class") != "Qwen3_5ForCausalLM":
        raise ValueError("G2 loaded the multimodal model instead of the text-only CausalLM")
    if run_manifest.get("model_config_type") != "qwen3_5_text" or run_manifest.get("vision_tower_loaded") is not False:
        raise ValueError("G2 text-only model contract drift")
    if run_manifest.get("schedule_validation", {}).get("schedule_sha256") != args.expected_schedule_sha256:
        raise ValueError("G2 schedule digest drift")
    if run_manifest.get("precision_policy") != {
        "parameters": "FP32",
        "gradients": "FP32",
        "adamw_moments": "FP32",
        "forward_backward_autocast": "BF16",
    }:
        raise ValueError("G2 precision policy drift")
    selective = run_manifest.get("selective_output_projection", {})
    if (
        selective.get("enabled") is not True
        or selective.get("pinned_liger_commit") != PINNED_LIGER_COMMIT
        or selective.get("implementation") != "pinned_liger_fused_linear_cross_entropy"
    ):
        raise ValueError("G2 selective fused-output projection was not proven")
    formula = dict(run_manifest.get("flop_formula", {}))
    formula_sha256 = formula.pop("formula_sha256", None)
    if formula.get("formula_version") != EXPECTED_FLOP_FORMULA_VERSION:
        raise ValueError("G2 Qwen3.5 FLOP formula version drift")
    recomputed_formula_sha256 = hashlib.sha256(
        json.dumps(formula, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if formula_sha256 != recomputed_formula_sha256:
        raise ValueError("G2 Qwen3.5 FLOP formula hash drift")
    formula_integer_fields = {
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_gdn_layers",
        "num_full_attention_layers",
        "full_attention_heads",
        "full_attention_kv_heads",
        "full_attention_head_dim",
        "gdn_heads",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
        "vocabulary_size",
        "decoder_linear_training_flops_per_fixed_token",
        "gdn_training_flops_per_fixed_token",
        "nominal_peak_flops_per_second_per_gpu",
    }
    if any(not isinstance(formula.get(key), int) or formula[key] <= 0 for key in formula_integer_fields):
        raise ValueError("G2 Qwen3.5 FLOP formula has missing or invalid integer fields")
    if formula["num_gdn_layers"] + formula["num_full_attention_layers"] != formula["num_layers"]:
        raise ValueError("G2 Qwen3.5 FLOP formula layer counts drift")
    hidden = formula["hidden_size"]
    gdn_key_width = formula["gdn_heads"] * formula["gdn_key_head_dim"]
    gdn_value_width = formula["gdn_heads"] * formula["gdn_value_head_dim"]
    mlp_weights = formula["num_layers"] * 3 * hidden * formula["intermediate_size"]
    full_weights_per_layer = (
        hidden * (2 * formula["full_attention_heads"] * formula["full_attention_head_dim"])
        + 2 * hidden * (formula["full_attention_kv_heads"] * formula["full_attention_head_dim"])
        + formula["full_attention_heads"] * formula["full_attention_head_dim"] * hidden
    )
    gdn_weights_per_layer = (
        hidden * (2 * gdn_key_width + gdn_value_width)
        + hidden * gdn_value_width
        + 2 * hidden * formula["gdn_heads"]
        + gdn_value_width * hidden
    )
    expected_decoder_coefficient = 6 * (
        mlp_weights
        + formula["num_full_attention_layers"] * full_weights_per_layer
        + formula["num_gdn_layers"] * gdn_weights_per_layer
    )
    expected_gdn_coefficient = (
        3
        * formula["num_gdn_layers"]
        * 7
        * formula["gdn_heads"]
        * formula["gdn_key_head_dim"]
        * formula["gdn_value_head_dim"]
    )
    if formula["decoder_linear_training_flops_per_fixed_token"] != expected_decoder_coefficient:
        raise ValueError("G2 decoder-linear FLOP coefficient drift")
    if formula["gdn_training_flops_per_fixed_token"] != expected_gdn_coefficient:
        raise ValueError("G2 GDN FLOP coefficient drift")
    expected_global_tokens = 32768 * args.expected_world_size * args.expected_gradient_accumulation
    if run_manifest.get("effective_tokens_per_optimizer_step") != expected_global_tokens:
        raise ValueError("G2 global sequence-token batch drift")
    if trainer_state.get("global_step") != args.expected_final_step:
        raise ValueError("root trainer state final step drift")
    if checkpoint_state.get("global_step") != args.expected_final_step:
        raise ValueError("checkpoint trainer state final step drift")
    if update_probe.get("status") != "passed":
        raise ValueError("parameter-update probe did not pass")
    if update_probe.get("observed_initial_global_step") != args.expected_initial_step:
        raise ValueError("parameter-update probe initial step drift")
    if update_probe.get("final_global_step") != args.expected_final_step:
        raise ValueError("parameter-update probe final step drift")
    if update_probe.get("optimizer_steps_observed") != args.expected_final_step - args.expected_initial_step:
        raise ValueError("parameter-update probe optimizer-step count drift")
    comparison = update_probe.get("parameter_comparison", {})
    if comparison.get("changed_sampled_values", 0) <= 0 or comparison.get("max_absolute_delta", 0) <= 0:
        raise ValueError("parameter-update probe contains no nonzero update")
    if not update_probe.get("finite_losses") or not update_probe.get("finite_gradient_norms"):
        raise ValueError("parameter-update probe lacks finite loss/gradient evidence")
    if conversion_ledger.get("target_class") != "Qwen3_5ForCausalLM":
        raise ValueError("conversion ledger target class drift")
    if conversion_ledger.get("tied_input_output_embeddings") is not True:
        raise ValueError("conversion ledger did not prove tied input/output embeddings")
    if conversion_ledger.get("tensor_hashes_enabled") is not True:
        raise ValueError("conversion ledger omitted parameter tensor hashes")
    if not conversion_ledger.get("rows") or not all(row.get("tensor_sha256") for row in conversion_ledger["rows"]):
        raise ValueError("conversion ledger contains an unhashed state tensor")

    expected_optimizer_steps = args.expected_final_step - args.expected_initial_step
    expected_group = args.expected_world_size * args.expected_gradient_accumulation
    expected_schedule_indices = list(
        range(args.expected_initial_step * expected_group, args.expected_final_step * expected_group)
    )
    observed_schedule_indices = [index for record in exact_metrics for index in record.get("schedule_indices", [])]
    if observed_schedule_indices != expected_schedule_indices:
        raise ValueError("exact metrics schedule exposure is not contiguous and no-repeat")
    expected_window_start = args.expected_initial_step + 1
    for record in exact_metrics:
        if record.get("schedule_sha256") != args.expected_schedule_sha256:
            raise ValueError("exact metrics schedule digest drift")
        optimizer_updates = int(record.get("optimizer_updates", 0))
        if optimizer_updates <= 0:
            raise ValueError("exact metrics window has no optimizer update")
        if record.get("window_start_step") != expected_window_start:
            raise ValueError("exact metrics windows are not contiguous")
        if record.get("step") != expected_window_start + optimizer_updates - 1:
            raise ValueError("exact metrics window endpoint drift")
        expected_window_start = int(record["step"]) + 1
        if record.get("synchronized_timing") is not True:
            raise ValueError("exact metrics timing was not synchronized")
        counts = record.get("counts", {})
        if counts.get("packs") != expected_group * optimizer_updates:
            raise ValueError("exact metrics global pack count drift")
        if counts.get("fixed_tokens") != 262_144 * optimizer_updates:
            raise ValueError("exact metrics global fixed-token count drift")
        if counts.get("real_tokens", -1) + counts.get("padding_tokens", -1) != counts["fixed_tokens"]:
            raise ValueError("exact metrics real/padding accounting drift")
        if record.get("loss", {}).get("global_assistant_target_divisor") != counts.get("assistant_targets"):
            raise ValueError("exact metrics loss divisor drift")
        expected_window_packs = expected_group * optimizer_updates
        if (
            len(record.get("pack_uids", [])) != expected_window_packs
            or len(set(record["pack_uids"])) != expected_window_packs
        ):
            raise ValueError("exact metrics pack identities are missing or repeated")
        elapsed = float(record.get("elapsed_seconds", 0))
        if elapsed <= 0:
            raise ValueError("exact metrics elapsed time is not positive")
        rates = record.get("rates", {})
        _close(
            float(rates.get("fixed_tokens_per_second_global", -1)),
            counts["fixed_tokens"] / elapsed,
            context="fixed-token rate",
        )
        _close(
            float(rates.get("fixed_tokens_per_second_per_gpu", -1)),
            counts["fixed_tokens"] / (args.expected_world_size * elapsed),
            context="per-GPU fixed-token rate",
        )
        _close(
            float(rates.get("real_tokens_per_second_global", -1)),
            counts["real_tokens"] / elapsed,
            context="real-token rate",
        )
        _close(
            float(rates.get("assistant_targets_per_second_global", -1)),
            counts["assistant_targets"] / elapsed,
            context="assistant-target rate",
        )
        _close(
            float(rates.get("optimizer_steps_per_second", -1)),
            optimizer_updates / elapsed,
            context="optimizer-step rate",
        )
        optimizer = record.get("optimizer", {})
        applied_learning_rates = optimizer.get("applied_learning_rates", [])
        if (
            len(applied_learning_rates) != optimizer_updates
            or optimizer.get("learning_rate") != applied_learning_rates[-1]
        ):
            raise ValueError("exact metrics applied learning-rate count drift")
        analytic = record.get("analytic_flops", {})
        if (
            analytic.get("formula_version") != EXPECTED_FLOP_FORMULA_VERSION
            or analytic.get("formula_sha256") != formula_sha256
        ):
            raise ValueError("exact metrics FLOP formula identity drift")
        expected_pairs = (counts["attention_length_squared"] + counts["fixed_tokens"]) // 2
        if (counts["attention_length_squared"] + counts["fixed_tokens"]) % 2 or analytic.get(
            "isolated_causal_attention_pairs"
        ) != expected_pairs:
            raise ValueError("exact metrics isolated causal-attention pair count drift")
        components = analytic.get("components", {})
        component_names = {
            "decoder_linear_and_mlp",
            "gdn_recurrence_approximation",
            "document_isolated_causal_full_attention",
            "selected_output_projection",
        }
        if any(not isinstance(components.get(name), int) or components[name] < 0 for name in component_names):
            raise ValueError("exact metrics contains an invalid FLOP component")
        expected_components = {
            "decoder_linear_and_mlp": expected_decoder_coefficient * counts["fixed_tokens"],
            "gdn_recurrence_approximation": expected_gdn_coefficient * counts["fixed_tokens"],
            "document_isolated_causal_full_attention": (
                12
                * formula["num_full_attention_layers"]
                * formula["full_attention_heads"]
                * formula["full_attention_head_dim"]
                * expected_pairs
            ),
            "selected_output_projection": (
                6 * formula["hidden_size"] * formula["vocabulary_size"] * counts["assistant_targets"]
            ),
        }
        if any(components[name] != expected_components[name] for name in component_names):
            raise ValueError("exact metrics FLOP component derivation drift")
        if components.get("total") != sum(components[name] for name in component_names):
            raise ValueError("exact metrics FLOP component total drift")
        peak = int(analytic.get("nominal_peak_flops_per_second_per_gpu", 0))
        if peak != formula.get("nominal_peak_flops_per_second_per_gpu"):
            raise ValueError("exact metrics nominal peak denominator drift")
        _close(
            float(analytic.get("analytic_model_mfu", -1)),
            components["total"] / (args.expected_world_size * peak * elapsed),
            context="analytic MFU",
        )
    if expected_window_start != args.expected_final_step + 1:
        raise ValueError("exact metrics do not reach the expected final step")
    if (
        exact_metrics_summary.get("first_step") != args.expected_initial_step + 1
        or exact_metrics_summary.get("last_step") != args.expected_final_step
    ):
        raise ValueError("exact metrics summary interval drift")
    if exact_metrics_summary.get("schedule_sha256") != args.expected_schedule_sha256:
        raise ValueError("exact metrics summary schedule drift")
    summary_counts = {
        key: sum(int(record["counts"][key]) for record in exact_metrics) for key in exact_metrics[0]["counts"]
    }
    if exact_metrics_summary.get("counts") != summary_counts:
        raise ValueError("exact metrics summary count aggregation drift")
    if exact_metrics_summary.get("optimizer_steps") != expected_optimizer_steps:
        raise ValueError("exact metrics summary optimizer-step count drift")
    if exact_metrics_summary.get("reporting_windows") != len(exact_metrics):
        raise ValueError("exact metrics summary window count drift")
    summary_elapsed = sum(float(record["elapsed_seconds"]) for record in exact_metrics)
    _close(float(exact_metrics_summary.get("elapsed_seconds", -1)), summary_elapsed, context="summary elapsed time")
    summary_rates = exact_metrics_summary.get("aggregate_rates", {})
    for rate_name, count_name in (
        ("fixed_tokens_per_second_global", "fixed_tokens"),
        ("real_tokens_per_second_global", "real_tokens"),
        ("assistant_targets_per_second_global", "assistant_targets"),
    ):
        _close(
            float(summary_rates.get(rate_name, -1)),
            summary_counts[count_name] / summary_elapsed,
            context=f"summary {rate_name}",
        )
    summary_flops = exact_metrics_summary.get("analytic_flops", {})
    total_flops = sum(record["analytic_flops"]["components"]["total"] for record in exact_metrics)
    if (
        summary_flops.get("formula_version") != EXPECTED_FLOP_FORMULA_VERSION
        or summary_flops.get("formula_sha256") != formula_sha256
        or summary_flops.get("total") != total_flops
    ):
        raise ValueError("exact metrics summary FLOP aggregation drift")
    _close(
        float(summary_flops.get("analytic_model_mfu", -1)),
        total_flops / (args.expected_world_size * formula["nominal_peak_flops_per_second_per_gpu"] * summary_elapsed),
        context="summary analytic MFU",
    )

    with safe_open(required["checkpoint_model"], framework="pt", device="cpu") as handle:
        bad_model_dtypes = {
            key: handle.get_slice(key).get_dtype()
            for key in handle.keys()  # noqa: SIM118 - safe_open is not iterable
            if handle.get_slice(key).get_dtype() != "F32"
        }
    if bad_model_dtypes:
        raise ValueError(f"checkpoint contains non-FP32 parameters: {list(bad_model_dtypes.items())[:10]}")
    optimizer_state = torch.load(required["checkpoint_optimizer"], map_location="cpu", weights_only=True)
    bad_optimizer_dtypes = []
    moment_tensors = 0
    for parameter_state in optimizer_state.get("state", {}).values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = parameter_state.get(key)
            if value is None:
                continue
            moment_tensors += 1
            if value.dtype != torch.float32:
                bad_optimizer_dtypes.append((key, str(value.dtype)))
    if moment_tensors == 0 or bad_optimizer_dtypes:
        raise ValueError(f"checkpoint AdamW moments are missing or non-FP32: {bad_optimizer_dtypes[:10]}")

    optional_reports: dict[str, Any] = {}
    if args.require_conversion_parity:
        path = root / "base_reference_and_conversion_parity.json"
        report = read_json(path)
        validate_h1_reference_report(report, expected_manifest_sha256=args.expected_qualification_manifest_sha256)
        optional_reports["conversion_parity_sha256"] = sha256_file(path)
    if args.require_selective_loss:
        path = root / "selective_liger_loss_qualification.json"
        report = read_json(path)
        patched = report.get("patched_qwen_forward", {})
        if (
            report.get("status") != "passed"
            or report.get("qualification_manifest_sha256") != args.expected_qualification_manifest_sha256
            or report.get("liger_kernel", {}).get("commit") != PINNED_LIGER_COMMIT
            or report.get("zero_target_sentinel", {}).get("loss") != 0
            or patched.get("model_class") != "Qwen3_5ForCausalLM"
            or "liger_kernel" not in patched.get("patched_forward_module", "")
            or patched.get("checked_parameter_gradients", 0) <= 0
        ):
            raise ValueError("selective Liger loss qualification did not pass")
        optional_reports["selective_loss_sha256"] = sha256_file(path)
    if args.require_ddp_normalization:
        path = root / "ddp_target_normalization_qualification.json"
        report = read_json(path)
        if (
            report.get("status") != "passed"
            or report.get("qualification_manifest_sha256") != args.expected_qualification_manifest_sha256
            or report.get("includes_zero_target_rank") is not True
        ):
            raise ValueError("DDP target normalization qualification did not pass")
        if report.get("world_size") != args.expected_world_size:
            raise ValueError("DDP normalization qualification world-size drift")
        optional_reports["ddp_normalization_sha256"] = sha256_file(path)
    if args.require_schedule_sharding:
        path = root / "accelerate_schedule_sharding_qualification.json"
        report = read_json(path)
        if (
            report.get("status") != "passed"
            or report.get("qualification_manifest_sha256") != args.expected_qualification_manifest_sha256
            or report.get("even_batches") is not False
        ):
            raise ValueError("Accelerate sequential-sharding qualification did not pass")
        if report.get("world_size") != args.expected_world_size:
            raise ValueError("Accelerate sharding qualification world-size drift")
        if report.get("global_indices_exactly_once") != list(range(report.get("schedule_length", -1))):
            raise ValueError("Accelerate sharding qualification duplicated or omitted an index")
        optional_reports["schedule_sharding_sha256"] = sha256_file(path)
    if args.require_hardware_profile:
        profile_path = root / "qwen35_cuda_hardware_profile.json"
        validation_path = root / "qwen35_h4_hardware_profile_validation.json"
        profile = read_json(profile_path)
        validation = read_json(validation_path)
        if (
            profile.get("status") != "captured_pending_kernel_audit"
            or profile.get("qualification_manifest_sha256") != args.expected_qualification_manifest_sha256
        ):
            raise ValueError("H4 raw hardware profile did not pass")
        if validation.get("status") != "required_categories_passed_pending_manual_kernel_source_review":
            raise ValueError("H4 automated profiler validation did not pass")
        if validation.get("qualification_manifest_sha256") != args.expected_qualification_manifest_sha256:
            raise ValueError("H4 profiler validation qualification-manifest drift")
        if validation.get("hardware_profile_sha256") != sha256_file(profile_path):
            raise ValueError("H4 profiler validation does not bind the raw profile")
        optional_reports["hardware_profile_sha256"] = sha256_file(profile_path)
        optional_reports["hardware_profile_validation_sha256"] = sha256_file(validation_path)
    if args.require_reference_parity:
        path = root / "resumed_reference_and_packed_parity.json"
        report = read_json(path)
        if (
            report.get("qualification_manifest_sha256") != args.expected_qualification_manifest_sha256
            or report.get("model_parity", {}).get("status") != "pass"
        ):
            raise ValueError("resumed packed-logit parity did not pass")
        if report.get("fixture_validation", {}).get("counts", {}).get("examples") != 128:
            raise ValueError("resumed reference parity did not cover the 128-case fixture")
        optional_reports["reference_parity_sha256"] = sha256_file(path)
    if args.require_generation_parser:
        path = root / "resumed_generation_parser_batch_invariance.json"
        report = read_json(path)
        if report.get("status") != "passed":
            raise ValueError("generation/parser report did not pass")
        parser_corpus = report.get("parser_corpus", {})
        if parser_corpus.get("invalid_cases_rejected") != parser_corpus.get("invalid_cases"):
            raise ValueError("generation/parser report did not reject every malformed case")
        generation = report.get("generation", {})
        if generation.get("single_batch_token_exact_for_all_cases") is not True:
            raise ValueError("generation single/batch token invariance failed")
        expected_case_ids = {
            "explicit_single_call",
            "explicit_parallel_calls",
            "sequential_second_call",
            "multi_turn_followup_call",
            "justified_no_call",
        }
        if {case.get("case_id") for case in generation.get("cases", [])} != expected_case_ids:
            raise ValueError("generation/parser semantic case coverage drift")
        optional_reports["generation_parser_sha256"] = sha256_file(path)
    if args.require_resume_parity:
        path = root / "continuous_resume_checkpoint_comparison.json"
        report = read_json(path)
        if report.get("status") != "passed":
            raise ValueError("continuous/resume checkpoint comparison did not pass")
        if report.get("atol") != 0 or report.get("rtol") != 0:
            raise ValueError("continuous/resume comparison was not zero tolerance")
        if report.get("model", {}).get("bit_exact") is not True:
            raise ValueError("continuous/resume model was not bit-exact")
        optional_reports["resume_parity_sha256"] = sha256_file(path)

    return {
        "artifact": "qwen35_g2_output_validation",
        "schema_version": 1,
        "status": "passed",
        "qualification_protocol_id": "qwen35-hardware-qualification-r15",
        "qualification_manifest_sha256": args.expected_qualification_manifest_sha256,
        "output_dir": str(root),
        "expected_initial_step": args.expected_initial_step,
        "expected_final_step": args.expected_final_step,
        "expected_world_size": args.expected_world_size,
        "expected_gradient_accumulation": args.expected_gradient_accumulation,
        "effective_tokens_per_optimizer_step": expected_global_tokens,
        "checkpoint_model_size_bytes": required["checkpoint_model"].stat().st_size,
        "input_report_sha256": {
            label: sha256_file(path) for label, path in required.items() if path.suffix == ".json"
        },
        **optional_reports,
    }


def main() -> int:
    args = parse_args()
    report = validate(args)
    output = args.output_dir.resolve() / "g2_artifact_validation.json"
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.incomplete.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
