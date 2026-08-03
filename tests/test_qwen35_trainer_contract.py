import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from scripts.train.qwen35.train_qwen35_sft import (
    DEFAULT_MODEL_REVISION,
    EXPECTED_AMENDED_SAMPLE_UIDS,
    EXPECTED_CHAT_TEMPLATE_SHA256,
    EXPECTED_CORE_OPERATIONS_SHA256,
    EXPECTED_FROZEN_DESIGN_SHA256,
    EXPECTED_RENDERER_AMENDMENT_ID,
    EXPECTED_RENDERER_AMENDMENT_SHA256,
    EXPECTED_SUITE_ID,
    DataArguments,
    Qwen35ExactMetricsCallback,
    Qwen35UpdateProbeCallback,
    compare_parameter_probe_samples,
    evenly_spaced_integer_indices,
    parameter_probe_samples,
    validate_frozen_data_contract,
    validate_saved_tokenizer,
)

from open_instruct.qwen35_reporting import Qwen35FlopFormula


def frozen_manifest() -> dict:
    return {
        "suite_id": EXPECTED_SUITE_ID,
        "arm_id": "C00",
        "renderer": "qwen35_native_tools",
        "enable_thinking": False,
        "max_seq_length": 32768,
        "packing_semantics": ("atomic_documents_best_fit_decreasing_no_cross_pack_or_part_splits"),
        "documents_index": "documents.jsonl.gz",
        "documents_index_sha256": "d" * 64,
        "tokenizer": {"revision": DEFAULT_MODEL_REVISION, "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256},
        "inputs": {
            "core_operations_sha256": EXPECTED_CORE_OPERATIONS_SHA256,
            "frozen_design_manifest_sha256": EXPECTED_FROZEN_DESIGN_SHA256,
        },
        "renderer_amendment": {
            "amendment_id": EXPECTED_RENDERER_AMENDMENT_ID,
            "manifest_sha256": EXPECTED_RENDERER_AMENDMENT_SHA256,
            "observed_sample_uids": sorted(EXPECTED_AMENDED_SAMPLE_UIDS),
        },
    }


def dataset(manifest: dict | None = None, accounting: dict | None = None):
    return SimpleNamespace(
        manifest=manifest or frozen_manifest(),
        accounting=lambda: (
            accounting
            or {"raw_tokens": 100, "packed_real_tokens": 100, "dropped_tokens": 0, "effective_trainable_tokens": 20}
        ),
    )


def data_arguments() -> DataArguments:
    return DataArguments(
        numpy_data_dir="/irrelevant",
        expected_arm_id="C00",
        pack_schedule_path="/irrelevant/schedule.json",
        expected_schedule_sha256="a" * 64,
        sequence_length=32768,
        verify_data_hashes=True,
        drop_last=False,
    )


def test_trainer_accepts_only_the_complete_frozen_c00_contract():
    report = validate_frozen_data_contract(data_arguments(), dataset())

    assert report["suite_id"] == EXPECTED_SUITE_ID
    assert report["arm_id"] == "C00"
    assert report["renderer_amendment_id"] == EXPECTED_RENDERER_AMENDMENT_ID
    assert set(report["amended_sample_uids"]) == EXPECTED_AMENDED_SAMPLE_UIDS
    assert report["packing_accounting"]["dropped_tokens"] == 0


@pytest.mark.parametrize(
    ("path", "bad_value", "message"),
    [
        (("suite_id",), "wrong", "suite ID"),
        (("arm_id",), "C01", "dataset arm"),
        (("enable_thinking",), True, "enable_thinking"),
        (("tokenizer", "revision"), "floating", "tokenizer revision"),
        (("renderer_amendment", "manifest_sha256"), "0" * 64, "manifest digest"),
        (("inputs", "core_operations_sha256"), "0" * 64, "operation ledger"),
    ],
)
def test_trainer_fails_closed_on_frozen_contract_drift(path, bad_value, message):
    manifest = deepcopy(frozen_manifest())
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value

    with pytest.raises(ValueError, match=message):
        validate_frozen_data_contract(data_arguments(), dataset(manifest=manifest))


def test_trainer_rejects_any_packer_token_loss():
    bad_accounting = {
        "raw_tokens": 100,
        "packed_real_tokens": 99,
        "dropped_tokens": 1,
        "effective_trainable_tokens": 20,
    }

    with pytest.raises(ValueError, match="dropped tokens"):
        validate_frozen_data_contract(data_arguments(), dataset(accounting=bad_accounting))


def test_saved_tokenizer_must_match_the_rendering_manifest():
    tokenizer_class = type(
        "Qwen2TokenizerFast",
        (),
        {"__len__": lambda self: 11, "vocab_size": 10, "chat_template": "template", "pad_token_id": 0},
    )
    tokenizer = tokenizer_class()
    manifest = frozen_manifest()
    manifest["tokenizer"].update(
        {
            "class": "Qwen2TokenizerFast",
            "vocab_size": 10,
            "length": 11,
            "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
        }
    )
    packed_dataset = dataset(manifest=manifest)

    report = validate_saved_tokenizer(tokenizer, packed_dataset)
    assert report["pad_token_id"] == 0

    tokenizer.chat_template = "tampered"
    with pytest.raises(ValueError, match="chat_template_sha256"):
        validate_saved_tokenizer(tokenizer, packed_dataset)


def test_sparse_parameter_probe_detects_a_real_finite_update():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    initial = parameter_probe_samples(model, parameter_limit=8, values_per_parameter=16)
    with torch.no_grad():
        model[0].weight.add_(0.25)
    final = parameter_probe_samples(model, parameter_limit=8, values_per_parameter=16)

    report = compare_parameter_probe_samples(initial, final)

    assert report["sampled_parameters"] == 4
    assert report["changed_sampled_values"] > 0
    assert report["max_absolute_delta"] == pytest.approx(0.25)


def test_evenly_spaced_integer_indices_are_exact_and_in_bounds_at_qwen35_geometry():
    # Qwen3.5-0.8B tied embedding/lm-head: 248,320 vocabulary rows x 1,024 hidden width.
    length = 248_320 * 1_024
    indices = evenly_spaced_integer_indices(length, 64)

    assert len(indices) == 64
    assert indices[0] == 0
    assert indices[-1] == length - 1
    assert all(left < right for left, right in zip(indices, indices[1:]))
    assert all(0 <= index < length for index in indices)
    assert indices == [(position * (length - 1)) // 63 for position in range(64)]


@pytest.mark.parametrize(
    ("length", "count", "expected"),
    [
        (1, 1, [0]),
        (2, 2, [0, 1]),
        (10, 5, [0, 2, 4, 6, 9]),
        (10, 10, list(range(10))),
    ],
)
def test_evenly_spaced_integer_indices_edge_cases(length, count, expected):
    assert evenly_spaced_integer_indices(length, count) == expected


@pytest.mark.parametrize(("length", "count"), [(0, 1), (-1, 1), (1, 0), (1, -1), (2, 3)])
def test_evenly_spaced_integer_indices_reject_invalid_bounds(length, count):
    with pytest.raises(ValueError, match="integer-index"):
        evenly_spaced_integer_indices(length, count)


def test_sparse_parameter_probe_rejects_no_update_and_nonfinite_values():
    model = torch.nn.Linear(2, 2)
    initial = parameter_probe_samples(model)

    with pytest.raises(RuntimeError, match="no sampled trainable parameter"):
        compare_parameter_probe_samples(initial, parameter_probe_samples(model))

    final = parameter_probe_samples(model)
    final["bias"]["values"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        compare_parameter_probe_samples(initial, final)


def test_update_callback_writes_complete_step_loss_gradient_and_weight_evidence(tmp_path):
    model = torch.nn.Linear(4, 2)
    state = SimpleNamespace(global_step=0, max_steps=1, is_world_process_zero=True)
    callback = Qwen35UpdateProbeCallback(tmp_path, expected_initial_global_step=0)
    callback.on_train_begin(None, state, None, model=model)
    callback.on_log(None, state, None, logs={"loss": 1.25, "grad_norm": 0.5})
    with torch.no_grad():
        model.weight.add_(0.125)
    state.global_step = 1

    callback.on_train_end(None, state, None, model=model)

    report = json.loads((tmp_path / "qwen35_parameter_update_probe.json").read_text())
    assert report["status"] == "passed"
    assert report["observed_initial_global_step"] == 0
    assert report["final_global_step"] == 1
    assert report["optimizer_steps_observed"] == 1
    assert report["finite_losses"] == [{"step": 0, "value": 1.25}]
    assert report["finite_gradient_norms"] == [{"step": 0, "value": 0.5}]
    assert report["parameter_comparison"]["changed_sampled_values"] > 0


def test_exact_metrics_cursor_maps_rank_and_microbatch_to_global_schedule_index(tmp_path):
    formula = Qwen35FlopFormula(
        hidden_size=1,
        intermediate_size=1,
        num_layers=1,
        num_gdn_layers=0,
        num_full_attention_layers=1,
        full_attention_heads=1,
        full_attention_kv_heads=1,
        full_attention_head_dim=1,
        gdn_heads=1,
        gdn_key_head_dim=1,
        gdn_value_head_dim=1,
        vocabulary_size=2,
        decoder_linear_training_flops_per_fixed_token=1,
        gdn_training_flops_per_fixed_token=0,
    )
    callback = Qwen35ExactMetricsCallback(
        output_dir=tmp_path,
        sequence_length=8,
        schedule_sha256="a" * 64,
        formula=formula,
        expected_initial_global_step=5,
        expected_final_global_step=6,
        sync_interval=1,
    )
    callback.configure_gradient_accumulation(2)
    callback._world_size = 4
    callback._process_index = 2

    assert callback.expected_schedule_index(global_step=5, micro_step=0) == 42
    assert callback.expected_schedule_index(global_step=5, micro_step=1) == 46


def test_exact_metrics_cursor_rejects_schedule_or_target_drift(tmp_path):
    formula = Qwen35FlopFormula(
        hidden_size=1,
        intermediate_size=1,
        num_layers=1,
        num_gdn_layers=0,
        num_full_attention_layers=1,
        full_attention_heads=1,
        full_attention_kv_heads=1,
        full_attention_head_dim=1,
        gdn_heads=1,
        gdn_key_head_dim=1,
        gdn_value_head_dim=1,
        vocabulary_size=2,
        decoder_linear_training_flops_per_fixed_token=1,
        gdn_training_flops_per_fixed_token=0,
    )
    callback = Qwen35ExactMetricsCallback(
        output_dir=tmp_path,
        sequence_length=8,
        schedule_sha256="a" * 64,
        formula=formula,
        expected_initial_global_step=0,
        expected_final_global_step=1,
        sync_interval=1,
    )
    callback.configure_gradient_accumulation(1)
    metadata = {
        "_qwen35_schedule_index": 1,
        "_qwen35_pack_uid": "pack",
        "_qwen35_synthetic": False,
        "_qwen35_real_tokens": 8,
        "_qwen35_assistant_targets": 2,
        "_qwen35_padding_tokens": 0,
        "_qwen35_attention_length_squared": 64,
        "_qwen35_document_count": 1,
    }

    with pytest.raises(RuntimeError, match="schedule exposure drift"):
        callback.record_microbatch(
            global_step=0,
            metadata_row=metadata,
            observed_assistant_targets=2,
            loss=torch.tensor(1.0),
            num_items_in_batch=2,
            elapsed_seconds=0.1,
        )
    metadata["_qwen35_schedule_index"] = 0
    with pytest.raises(RuntimeError, match="selective-row count"):
        callback.record_microbatch(
            global_step=0,
            metadata_row=metadata,
            observed_assistant_targets=1,
            loss=torch.tensor(1.0),
            num_items_in_batch=2,
            elapsed_seconds=0.1,
        )


def test_exact_metrics_captures_the_learning_rate_before_scheduler_mutation(tmp_path):
    formula = Qwen35FlopFormula(
        hidden_size=1,
        intermediate_size=1,
        num_layers=1,
        num_gdn_layers=0,
        num_full_attention_layers=1,
        full_attention_heads=1,
        full_attention_kv_heads=1,
        full_attention_head_dim=1,
        gdn_heads=1,
        gdn_key_head_dim=1,
        gdn_value_head_dim=1,
        vocabulary_size=2,
        decoder_linear_training_flops_per_fixed_token=1,
        gdn_training_flops_per_fixed_token=0,
    )
    callback = Qwen35ExactMetricsCallback(
        output_dir=tmp_path,
        sequence_length=8,
        schedule_sha256="a" * 64,
        formula=formula,
        expected_initial_global_step=0,
        expected_final_global_step=1,
        sync_interval=1,
    )
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)

    callback.on_pre_optimizer_step(None, None, None, optimizer=optimizer)
    optimizer.param_groups[0]["lr"] = 1e-5

    assert callback.window_applied_learning_rates == [2e-5]


def test_exact_metrics_requires_and_recomputes_every_forward_selected_output_audit(tmp_path):
    formula = Qwen35FlopFormula(
        hidden_size=1024,
        intermediate_size=1,
        num_layers=1,
        num_gdn_layers=0,
        num_full_attention_layers=1,
        full_attention_heads=1,
        full_attention_kv_heads=1,
        full_attention_head_dim=1,
        gdn_heads=1,
        gdn_key_head_dim=1,
        gdn_value_head_dim=1,
        vocabulary_size=248320,
        decoder_linear_training_flops_per_fixed_token=1,
        gdn_training_flops_per_fixed_token=0,
    )
    callback = Qwen35ExactMetricsCallback(
        output_dir=tmp_path,
        sequence_length=32768,
        schedule_sha256="a" * 64,
        formula=formula,
        expected_initial_global_step=0,
        expected_final_global_step=1,
        sync_interval=1,
    )
    callback.configure_gradient_accumulation(1)
    callback.configure_forward_loss_audit(
        required=True,
        chunk_size=128,
        vocabulary_size=248320,
        hidden_size=1024,
    )
    metadata = {
        "_qwen35_schedule_index": 0,
        "_qwen35_pack_uid": "pack",
        "_qwen35_synthetic": False,
        "_qwen35_real_tokens": 32768,
        "_qwen35_assistant_targets": 129,
        "_qwen35_padding_tokens": 0,
        "_qwen35_attention_length_squared": 32768**2,
        "_qwen35_document_count": 1,
    }
    audit = {
        "checkpointed": True,
        "chunk_boundaries": [[0, 128], [128, 129]],
        "chunk_count": 2,
        "chunk_size": 128,
        "full_selected_logit_elements": 129 * 248320,
        "global_target_count": 129,
        "hidden_size": 1024,
        "implementation_id": "pytorch_nonreentrant_checkpointed_chunked_selected_rows_r1",
        "maximum_chunk_rows": 128,
        "maximum_logit_elements": 128 * 248320,
        "returned_dense_logits": False,
        "selected_rows": 129,
        "vocabulary_size": 248320,
        "zero_target": False,
    }
    callback.record_microbatch(
        global_step=0,
        metadata_row=metadata,
        observed_assistant_targets=129,
        loss=torch.tensor(1.0),
        num_items_in_batch=129,
        elapsed_seconds=0.1,
        loss_audit=audit,
    )
    assert callback.window_selected_output_audits[0]["audit"] == audit

    missing = Qwen35ExactMetricsCallback(
        output_dir=tmp_path / "missing",
        sequence_length=32768,
        schedule_sha256="a" * 64,
        formula=formula,
        expected_initial_global_step=0,
        expected_final_global_step=1,
        sync_interval=1,
    )
    missing.configure_gradient_accumulation(1)
    missing.configure_forward_loss_audit(
        required=True,
        chunk_size=128,
        vocabulary_size=248320,
        hidden_size=1024,
    )
    with pytest.raises(RuntimeError, match="emitted no loss audit"):
        missing.record_microbatch(
            global_step=0,
            metadata_row=metadata,
            observed_assistant_targets=129,
            loss=torch.tensor(1.0),
            num_items_in_batch=129,
            elapsed_seconds=0.1,
            loss_audit=None,
        )
