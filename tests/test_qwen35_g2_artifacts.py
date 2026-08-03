import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from scripts.train.qwen35.validate_qwen35_g2_artifacts import PINNED_LIGER_COMMIT, validate

from open_instruct.qwen35_qualification import EVIDENCE_SERIALIZATION_CONTRACT

SCHEDULE_SHA256 = "a" * 64
QUALIFICATION_MANIFEST_SHA256 = "d" * 64
H1_VOCABULARY_SIZE = 248320
H1_HIDDEN_SIZE = 1024
H1_LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 6
H1_STATE_TENSORS = 321
H1_STATE_NUMEL = 1_006_672_704


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def exact_h1_metrics(*, shape=(8, H1_VOCABULARY_SIZE), atol=0, rtol=0):
    elements = math.prod(shape)
    return {
        "shape": list(shape),
        "elements": elements,
        "bit_exact": True,
        "allclose": True,
        "nonfinite_count": 0,
        "mismatched_elements": 0,
        "mismatched_fraction": 0.0,
        "maximum_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "absolute_error_quantiles": {"p50": 0.0, "p90": 0.0, "p99": 0.0, "p99_9": 0.0},
        "relative_l2_error": 0.0,
        "cosine_similarity": 1.0,
        "observed_l2_norm": 1.0,
        "reference_l2_norm": 1.0,
        "difference_l2_norm": 0.0,
        "top1_agreement": 1.0,
        "atol": atol,
        "rtol": rtol,
    }


def nonexact_h1_metrics(*, shape=(8, H1_VOCABULARY_SIZE), atol=0, rtol=0):
    value = exact_h1_metrics(shape=shape, atol=atol, rtol=rtol)
    elements = math.prod(shape)
    value.update(
        {
            "bit_exact": False,
            "allclose": False,
            "mismatched_elements": 1,
            "mismatched_fraction": 1 / elements,
            "maximum_absolute_error": 0.375,
            "mean_absolute_error": 0.375 / elements,
            "relative_l2_error": 0.01,
            "cosine_similarity": 0.9999,
            "difference_l2_norm": 0.375,
            "top1_agreement": 0.875,
        }
    )
    return value


def h1_layer_rows(length, *, diagnostic=False, first_nonexact=False):
    rows = []
    for layer_index, layer_type in enumerate(H1_LAYER_TYPES):
        kwargs = {
            "shape": (length, H1_HIDDEN_SIZE),
            "atol": 0.05 if diagnostic else 0,
            "rtol": 0.01 if diagnostic else 0,
        }
        metrics = nonexact_h1_metrics(**kwargs) if first_nonexact and layer_index == 0 else exact_h1_metrics(**kwargs)
        rows.append({"layer_index": layer_index, "layer_type": layer_type, "metrics": metrics})
    return rows


def complete_h1_report():
    state_rows = []
    for index in range(H1_STATE_TENSORS):
        numel = H1_STATE_NUMEL - (H1_STATE_TENSORS - 1) if index == 0 else 1
        state_rows.append(
            {
                "target_key": f"model.tensor_{index}",
                "source_key": f"model.language_model.tensor_{index}",
                "shape": [numel],
                "source_shape": [numel],
                "dtype": "torch.bfloat16",
                "source_dtype": "torch.bfloat16",
                "numel": numel,
                "target_sha256": "e" * 64,
                "source_sha256": "e" * 64,
                "bit_exact": True,
            }
        )
    state_rows_sha256 = hashlib.sha256(
        json.dumps(state_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    layer_rows_8 = h1_layer_rows(8)
    layer_rows_9 = h1_layer_rows(9)
    diagnostic_layer_rows_8 = h1_layer_rows(8, diagnostic=True)
    diagnostic_layer_rows_9 = h1_layer_rows(9, diagnostic=True)
    return {
        "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA256,
        "model_parity": {
            "status": "pass",
            "failures": [],
            "source_config_model_type": "qwen3_5",
            "production_model_class": "Qwen3_5ForCausalLM",
            "vocabulary_size": H1_VOCABULARY_SIZE,
            "text_hidden_size": H1_HIDDEN_SIZE,
            "text_num_hidden_layers": len(H1_LAYER_TYPES),
            "text_layer_types": H1_LAYER_TYPES,
            "standalone_reference_definition": (
                "one document executed with the exact production packed metadata/kernel path"
            ),
            "atol": 0.05,
            "rtol": 0.01,
            "sequence_lengths": [8, 9],
            "conditional_to_text_conversion": {
                "checked": True,
                "status": "pass",
                "atol": 0,
                "rtol": 0,
                "loading_info": {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
                "loading_info_serialization": EVIDENCE_SERIALIZATION_CONTRACT,
                "state_mapping": {
                    "status": "pass",
                    "target_tensor_count": H1_STATE_TENSORS,
                    "target_state_numel": H1_STATE_NUMEL,
                    "mismatched_target_keys": [],
                    "rows_sha256": state_rows_sha256,
                    "rows": state_rows,
                },
                "ordinary_logit_metrics": [
                    exact_h1_metrics(shape=(8, H1_VOCABULARY_SIZE)),
                    exact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE)),
                ],
                "dense_next_token_losses": [
                    {"causal": 1.0, "conditional": 1.0, "finite": True, "bit_exact": True, "absolute_error": 0.0},
                    {"causal": 2.0, "conditional": 2.0, "finite": True, "bit_exact": True, "absolute_error": 0.0},
                ],
            },
            "singleton_vs_multi_pack_shape_diagnostic": {
                "gating": False,
                "reason": "packed launch shape changes",
                "frozen_r9_tolerance_observation": "exceeds_tolerance",
                "r11_failed_criterion_reclassified_as_pass": False,
                "logits": [
                    nonexact_h1_metrics(shape=(8, H1_VOCABULARY_SIZE), atol=0.05, rtol=0.01),
                    nonexact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE), atol=0.05, rtol=0.01),
                ],
                "layers": [diagnostic_layer_rows_8, diagnostic_layer_rows_9],
            },
            "cross_kernel_ordinary_vs_singleton_diagnostic": {
                "gating": False,
                "reason": "different kernel paths",
                "logits": [
                    exact_h1_metrics(shape=(8, H1_VOCABULARY_SIZE), atol=0.05, rtol=0.01),
                    exact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE), atol=0.05, rtol=0.01),
                ],
                "layers": [diagnostic_layer_rows_8, diagnostic_layer_rows_9],
            },
            "single_token_counterfactual_no_cross_document_influence": {
                "mutate_first_hold_second": {
                    "mutation_position": 4,
                    "unchanged_segment_logits": exact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE)),
                    "unchanged_segment_layers": layer_rows_9,
                },
                "mutate_second_hold_first": {
                    "mutation_position": 4,
                    "unchanged_segment_logits": exact_h1_metrics(),
                    "unchanged_segment_layers": layer_rows_8,
                },
            },
            "full_document_counterfactual_no_cross_document_influence": {
                "mutate_every_first_token_hold_second": {
                    "mutated_tokens": 8,
                    "unchanged_segment_logits": exact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE)),
                    "unchanged_segment_layers": layer_rows_9,
                },
                "mutate_every_second_token_hold_first": {
                    "mutated_tokens": 9,
                    "unchanged_segment_logits": exact_h1_metrics(),
                    "unchanged_segment_layers": layer_rows_8,
                },
            },
            "duplicate_document_reset_invariance": {
                "sequence_0_first_vs_second": {
                    "unchanged_segment_logits": exact_h1_metrics(),
                    "unchanged_segment_layers": layer_rows_8,
                },
                "sequence_1_first_vs_second": {
                    "unchanged_segment_logits": exact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE)),
                    "unchanged_segment_layers": layer_rows_9,
                },
            },
            "packed_order_invariance": {
                "first_document_moved_to_second": {
                    "unchanged_segment_logits": exact_h1_metrics(),
                    "unchanged_segment_layers": layer_rows_8,
                },
                "second_document_moved_to_first": {
                    "unchanged_segment_logits": exact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE)),
                    "unchanged_segment_layers": layer_rows_9,
                },
            },
            "corrupted_boundary_negative_control": {
                "expected_bit_exact": False,
                "sensitivity_passed": True,
                "unchanged_segment_logits": nonexact_h1_metrics(shape=(9, H1_VOCABULARY_SIZE)),
                "unchanged_segment_layers": h1_layer_rows(9, first_nonexact=True),
            },
        },
    }


def complete_output(tmp_path):
    root = tmp_path / "g2-c"
    formula = {
        "hidden_size": 1,
        "intermediate_size": 2,
        "num_layers": 2,
        "num_gdn_layers": 1,
        "num_full_attention_layers": 1,
        "full_attention_heads": 1,
        "full_attention_kv_heads": 1,
        "full_attention_head_dim": 1,
        "gdn_heads": 1,
        "gdn_key_head_dim": 1,
        "gdn_value_head_dim": 1,
        "vocabulary_size": 10,
        "decoder_linear_training_flops_per_fixed_token": 144,
        "gdn_training_flops_per_fixed_token": 21,
        "formula_version": "qwen35-hybrid-causal-selected-output-v2",
        "nominal_peak_flops_per_second_per_gpu": 312_000_000_000_000,
    }
    formula["formula_sha256"] = hashlib.sha256(
        json.dumps(formula, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    run_manifest = {
        "frozen_data_validation": {
            "arm_id": "C00",
            "runtime_tokenizer_validation": {
                "class": "Qwen2TokenizerFast",
                "chat_template_sha256": "c" * 64,
                "pad_token_id": 0,
            },
        },
        "sequence_length": 32768,
        "drop_last": False,
        "data_hash_verification": "identity_bearing_numpy_files_on_global_rank_0_before_model_load",
        "world_size": 4,
        "gradient_accumulation_steps": 2,
        "effective_tokens_per_optimizer_step": 262144,
        "model_class": "Qwen3_5ForCausalLM",
        "model_config_type": "qwen3_5_text",
        "vision_tower_loaded": False,
        "schedule_validation": {"schedule_sha256": SCHEDULE_SHA256},
        "precision_policy": {
            "parameters": "FP32",
            "gradients": "FP32",
            "adamw_moments": "FP32",
            "forward_backward_autocast": "BF16",
        },
        "selective_output_projection": {
            "enabled": True,
            "implementation": "pinned_liger_fused_linear_cross_entropy",
            "pinned_liger_commit": PINNED_LIGER_COMMIT,
        },
        "flop_formula": formula,
        "hardware_qualification": {
            "protocol_id": "qwen35-hardware-qualification-r15",
            "manifest_path": "/frozen/qwen35_hardware_qualification_r15.json",
            "manifest_sha256": QUALIFICATION_MANIFEST_SHA256,
            "require_no_dense_logits": True,
            "hardware_profile": False,
            "cuda_event_step_timing": False,
        },
    }
    update_probe = {
        "status": "passed",
        "observed_initial_global_step": 5,
        "final_global_step": 10,
        "optimizer_steps_observed": 5,
        "finite_losses": [{"step": 6, "value": 1.0}],
        "finite_gradient_norms": [{"step": 6, "value": 0.5}],
        "parameter_comparison": {"changed_sampled_values": 4, "max_absolute_delta": 0.125},
    }
    trainer_state = {"global_step": 10, "log_history": [{"loss": 1.0}]}
    write_json(root / "qwen35_run_manifest.json", run_manifest)
    write_json(root / "qwen35_parameter_update_probe.json", update_probe)
    write_json(
        root / "qwen35_text_conversion_ledger.json",
        {
            "target_class": "Qwen3_5ForCausalLM",
            "tied_input_output_embeddings": True,
            "tensor_hashes_enabled": True,
            "rows": [{"target_key": "model.weight", "tensor_sha256": "b" * 64}],
        },
    )
    exact_metrics = []
    for step in range(6, 11):
        start = (step - 1) * 8
        counts = {
            "fixed_tokens": 262144,
            "real_tokens": 240000,
            "assistant_targets": 100,
            "padding_tokens": 22144,
            "attention_length_squared": 1_000_000,
            "documents": 16,
            "packs": 8,
            "synthetic_packs": 0,
        }
        elapsed = 2.0
        causal_pairs = (counts["attention_length_squared"] + counts["fixed_tokens"]) // 2
        components = {
            "decoder_linear_and_mlp": 144 * counts["fixed_tokens"],
            "gdn_recurrence_approximation": 21 * counts["fixed_tokens"],
            "document_isolated_causal_full_attention": 12 * causal_pairs,
            "selected_output_projection": 6 * 10 * counts["assistant_targets"],
        }
        components["total"] = sum(components.values())
        exact_metrics.append(
            {
                "step": step,
                "window_start_step": step,
                "optimizer_updates": 1,
                "schedule_sha256": SCHEDULE_SHA256,
                "schedule_indices": list(range(start, start + 8)),
                "pack_uids": [f"pack-{index}" for index in range(start, start + 8)],
                "synchronized_timing": True,
                "elapsed_seconds": elapsed,
                "counts": counts,
                "rates": {
                    "fixed_tokens_per_second_global": counts["fixed_tokens"] / elapsed,
                    "fixed_tokens_per_second_per_gpu": counts["fixed_tokens"] / (4 * elapsed),
                    "real_tokens_per_second_global": counts["real_tokens"] / elapsed,
                    "assistant_targets_per_second_global": counts["assistant_targets"] / elapsed,
                    "optimizer_steps_per_second": 1 / elapsed,
                },
                "loss": {"global_assistant_target_divisor": 100},
                "optimizer": {"learning_rate": 2e-5, "applied_learning_rates": [2e-5]},
                "analytic_flops": {
                    "formula_version": formula["formula_version"],
                    "formula_sha256": formula["formula_sha256"],
                    "isolated_causal_attention_pairs": causal_pairs,
                    "components": components,
                    "nominal_peak_flops_per_second_per_gpu": formula["nominal_peak_flops_per_second_per_gpu"],
                    "analytic_model_mfu": components["total"]
                    / (4 * formula["nominal_peak_flops_per_second_per_gpu"] * elapsed),
                },
            }
        )
    (root / "qwen35_exact_metrics.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "qwen35_exact_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in exact_metrics))
    write_json(
        root / "qwen35_exact_metrics_summary.json",
        {
            "first_step": 6,
            "last_step": 10,
            "optimizer_steps": 5,
            "reporting_windows": 5,
            "elapsed_seconds": 10.0,
            "schedule_sha256": SCHEDULE_SHA256,
            "counts": {key: value * 5 for key, value in exact_metrics[0]["counts"].items()},
            "aggregate_rates": {
                "fixed_tokens_per_second_global": exact_metrics[0]["counts"]["fixed_tokens"] * 5 / 10,
                "real_tokens_per_second_global": exact_metrics[0]["counts"]["real_tokens"] * 5 / 10,
                "assistant_targets_per_second_global": exact_metrics[0]["counts"]["assistant_targets"] * 5 / 10,
            },
            "analytic_flops": {
                "formula_version": formula["formula_version"],
                "formula_sha256": formula["formula_sha256"],
                "total": exact_metrics[0]["analytic_flops"]["components"]["total"] * 5,
                "analytic_model_mfu": (
                    exact_metrics[0]["analytic_flops"]["components"]["total"]
                    * 5
                    / (4 * formula["nominal_peak_flops_per_second_per_gpu"] * 10)
                ),
            },
        },
    )
    write_json(root / "trainer_state.json", trainer_state)
    write_json(root / "train_results.json", {"train_loss": 1.0})
    write_json(root / "checkpoint-10/trainer_state.json", trainer_state)
    save_file({"model.weight": torch.ones(2, dtype=torch.float32)}, root / "checkpoint-10/model.safetensors")
    torch.save(
        {
            "state": {0: {"step": torch.tensor(1.0), "exp_avg": torch.ones(2), "exp_avg_sq": torch.ones(2)}},
            "param_groups": [],
        },
        root / "checkpoint-10/optimizer.pt",
    )
    torch.save({}, root / "checkpoint-10/scheduler.pt")
    for rank in range(4):
        torch.save({}, root / f"checkpoint-10/rng_state_{rank}.pth")
    write_json(
        root / "resumed_reference_and_packed_parity.json",
        {
            "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA256,
            "model_parity": {"status": "pass"},
            "fixture_validation": {"counts": {"examples": 128}},
        },
    )
    write_json(
        root / "resumed_generation_parser_batch_invariance.json",
        {
            "status": "passed",
            "parser_corpus": {"invalid_cases": 5, "invalid_cases_rejected": 5},
            "generation": {
                "single_batch_token_exact_for_all_cases": True,
                "cases": [
                    {"case_id": "explicit_single_call"},
                    {"case_id": "explicit_parallel_calls"},
                    {"case_id": "sequential_second_call"},
                    {"case_id": "multi_turn_followup_call"},
                    {"case_id": "justified_no_call"},
                ],
            },
        },
    )
    write_json(
        root / "continuous_resume_checkpoint_comparison.json",
        {"status": "passed", "atol": 0, "rtol": 0, "model": {"bit_exact": True}},
    )
    write_json(
        root / "selective_liger_loss_qualification.json",
        {
            "status": "passed",
            "qualification_manifest_sha256": QUALIFICATION_MANIFEST_SHA256,
            "liger_kernel": {"commit": PINNED_LIGER_COMMIT},
            "zero_target_sentinel": {"loss": 0},
            "patched_qwen_forward": {
                "model_class": "Qwen3_5ForCausalLM",
                "patched_forward_module": "liger_kernel.transformers.model.qwen3_5",
                "checked_parameter_gradients": 18,
            },
        },
    )
    return root


def arguments(root):
    return SimpleNamespace(
        output_dir=root,
        expected_initial_step=5,
        expected_final_step=10,
        expected_world_size=4,
        expected_gradient_accumulation=2,
        expected_schedule_sha256=SCHEDULE_SHA256,
        expected_qualification_manifest_sha256=QUALIFICATION_MANIFEST_SHA256,
        require_reference_parity=True,
        require_generation_parser=True,
        require_resume_parity=True,
        require_conversion_parity=False,
        require_selective_loss=False,
        require_ddp_normalization=False,
        require_schedule_sharding=False,
        require_hardware_profile=False,
    )


def test_complete_g2_c_output_passes_every_machine_readable_gate(tmp_path):
    root = complete_output(tmp_path)

    report = validate(arguments(root))

    assert report["status"] == "passed"
    assert report["effective_tokens_per_optimizer_step"] == 262144
    assert report["checkpoint_model_size_bytes"] > 0
    assert "reference_parity_sha256" in report
    assert "generation_parser_sha256" in report
    assert "resume_parity_sha256" in report


def test_g2_output_validator_rejects_step_batch_update_and_parser_drift(tmp_path):
    root = complete_output(tmp_path)
    args = arguments(root)

    run_manifest = json.loads((root / "qwen35_run_manifest.json").read_text())
    run_manifest["world_size"] = 1
    write_json(root / "qwen35_run_manifest.json", run_manifest)
    with pytest.raises(ValueError, match="world size"):
        validate(args)

    root = complete_output(tmp_path / "second")
    args = arguments(root)
    update = json.loads((root / "qwen35_parameter_update_probe.json").read_text())
    update["parameter_comparison"]["changed_sampled_values"] = 0
    write_json(root / "qwen35_parameter_update_probe.json", update)
    with pytest.raises(ValueError, match="no nonzero update"):
        validate(args)

    root = complete_output(tmp_path / "third")
    args = arguments(root)
    generation = json.loads((root / "resumed_generation_parser_batch_invariance.json").read_text())
    generation["generation"]["cases"].pop()
    write_json(root / "resumed_generation_parser_batch_invariance.json", generation)
    with pytest.raises(ValueError, match="case coverage"):
        validate(args)

    root = complete_output(tmp_path / "fourth")
    args = arguments(root)
    metrics_path = root / "qwen35_exact_metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    metrics[0]["analytic_flops"]["analytic_model_mfu"] *= 2
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in metrics))
    with pytest.raises(ValueError, match="analytic MFU"):
        validate(args)


def test_g2_output_validator_requires_pinned_integrated_selective_loss_proof(tmp_path):
    root = complete_output(tmp_path)
    args = arguments(root)
    args.require_selective_loss = True

    report = validate(args)
    assert "selective_loss_sha256" in report

    selective = json.loads((root / "selective_liger_loss_qualification.json").read_text())
    selective["patched_qwen_forward"]["checked_parameter_gradients"] = 0
    write_json(root / "selective_liger_loss_qualification.json", selective)
    with pytest.raises(ValueError, match="selective Liger"):
        validate(args)


def test_g2_output_validator_independently_rechecks_complete_h1_evidence(tmp_path):
    root = complete_output(tmp_path)
    args = arguments(root)
    args.require_conversion_parity = True
    write_json(root / "base_reference_and_conversion_parity.json", complete_h1_report())

    report = validate(args)
    assert "conversion_parity_sha256" in report

    invalid = complete_h1_report()
    invalid["model_parity"]["single_token_counterfactual_no_cross_document_influence"]["mutate_first_hold_second"][
        "unchanged_segment_layers"
    ][0]["metrics"]["bit_exact"] = False
    write_json(root / "base_reference_and_conversion_parity.json", invalid)
    with pytest.raises(ValueError, match="counterfactual.*not finite and bit-exact"):
        validate(args)

    invalid = complete_h1_report()
    invalid["model_parity"]["corrupted_boundary_negative_control"]["sensitivity_passed"] = False
    write_json(root / "base_reference_and_conversion_parity.json", invalid)
    with pytest.raises(ValueError, match="negative control did not pass"):
        validate(args)


def test_g2_output_validator_rejects_self_asserted_h1_state_mapping(tmp_path):
    root = complete_output(tmp_path)
    args = arguments(root)
    args.require_conversion_parity = True
    invalid = complete_h1_report()
    invalid["model_parity"]["conditional_to_text_conversion"]["state_mapping"]["rows"][0]["source_sha256"] = "f" * 64
    write_json(root / "base_reference_and_conversion_parity.json", invalid)

    with pytest.raises(ValueError, match="state tensor.*not hash-identical"):
        validate(args)


@pytest.mark.parametrize("corruption", ["truncated_logits", "missing_layer", "truncated_state"])
def test_g2_output_validator_rejects_truncated_h1_coverage(tmp_path, corruption):
    root = complete_output(tmp_path)
    args = arguments(root)
    args.require_conversion_parity = True
    invalid = complete_h1_report()
    parity = invalid["model_parity"]
    if corruption == "truncated_logits":
        parity["full_document_counterfactual_no_cross_document_influence"]["mutate_every_first_token_hold_second"][
            "unchanged_segment_logits"
        ]["shape"][0] -= 1
    elif corruption == "missing_layer":
        parity["duplicate_document_reset_invariance"]["sequence_0_first_vs_second"]["unchanged_segment_layers"].pop()
    else:
        state_mapping = parity["conditional_to_text_conversion"]["state_mapping"]
        state_mapping["rows"].pop()
        state_mapping["target_tensor_count"] -= 1
    write_json(root / "base_reference_and_conversion_parity.json", invalid)

    with pytest.raises(ValueError, match="geometry drift|coverage drift"):
        validate(args)


def test_g2_output_validator_accepts_one_exact_multiupdate_window(tmp_path):
    root = complete_output(tmp_path)
    path = root / "qwen35_exact_metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    coarse = json.loads(json.dumps(records[-1]))
    coarse["window_start_step"] = records[0]["window_start_step"]
    coarse["optimizer_updates"] = len(records)
    coarse["schedule_indices"] = [index for record in records for index in record["schedule_indices"]]
    coarse["pack_uids"] = [uid for record in records for uid in record["pack_uids"]]
    coarse["elapsed_seconds"] = sum(record["elapsed_seconds"] for record in records)
    coarse["counts"] = {key: sum(record["counts"][key] for record in records) for key in records[0]["counts"]}
    coarse["rates"] = {
        "fixed_tokens_per_second_global": coarse["counts"]["fixed_tokens"] / coarse["elapsed_seconds"],
        "fixed_tokens_per_second_per_gpu": coarse["counts"]["fixed_tokens"] / (4 * coarse["elapsed_seconds"]),
        "real_tokens_per_second_global": coarse["counts"]["real_tokens"] / coarse["elapsed_seconds"],
        "assistant_targets_per_second_global": coarse["counts"]["assistant_targets"] / coarse["elapsed_seconds"],
        "optimizer_steps_per_second": len(records) / coarse["elapsed_seconds"],
    }
    coarse["loss"]["global_assistant_target_divisor"] = coarse["counts"]["assistant_targets"]
    coarse["optimizer"]["applied_learning_rates"] = [
        rate for record in records for rate in record["optimizer"]["applied_learning_rates"]
    ]
    coarse["optimizer"]["learning_rate"] = coarse["optimizer"]["applied_learning_rates"][-1]
    coarse["analytic_flops"]["isolated_causal_attention_pairs"] = (
        coarse["counts"]["attention_length_squared"] + coarse["counts"]["fixed_tokens"]
    ) // 2
    for key in records[0]["analytic_flops"]["components"]:
        coarse["analytic_flops"]["components"][key] = sum(
            record["analytic_flops"]["components"][key] for record in records
        )
    formula = json.loads((root / "qwen35_run_manifest.json").read_text())["flop_formula"]
    coarse["analytic_flops"]["analytic_model_mfu"] = coarse["analytic_flops"]["components"]["total"] / (
        4 * formula["nominal_peak_flops_per_second_per_gpu"] * coarse["elapsed_seconds"]
    )
    path.write_text(json.dumps(coarse) + "\n")
    summary_path = root / "qwen35_exact_metrics_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["reporting_windows"] = 1
    write_json(summary_path, summary)

    report = validate(arguments(root))

    assert report["status"] == "passed"
