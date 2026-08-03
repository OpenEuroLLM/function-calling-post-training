from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.train.qwen35.validate_qwen35_h5_checkpoint_reload_r18 import normalize_loading_info
from scripts.train.qwen35.validate_qwen35_h5_nccl_trace_r18 import sanitize_profiler_trace, validate_trace
from scripts.train.qwen35.validate_qwen35_h5_r18 import (
    EXPECTED_EXPOSURE_TOTALS,
    NCCL_FAILURE_PATTERN,
    _expected_learning_rates,
    _validate_schedule,
)

from open_instruct.qwen35_qualification_r18_h4 import load_strict_json, sha256_file
from open_instruct.qwen35_qualification_r18_h5 import (
    H5_CONTRACT_SHA256,
    H5_EXPECTED_TARGETS_BY_UPDATE,
    H5_FIRST_FIVE_ENTRIES_SHA256,
    H5_HARNESS_AMENDMENT_R2_SHA256,
    H5_HARNESS_AMENDMENT_SHA256,
    H5_HUMAN_PROTOCOL_SHA256,
    H5_PREREGISTRATION_CLOSURE_SHA256,
    H5_SELECTED_CHUNK_SIZE,
    load_h5_contract,
    load_h5_harness_amendment,
    load_h5_harness_amendment_r2,
    validate_h5_source_delta,
)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
CONTRACT = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h5.json"
WRAPPER = ROOT / "scripts/train/qwen35/leonardo_h5_r18.sbatch"
HARNESS_AMENDMENT = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h5_harness_amendment_r1.json"
HARNESS_AMENDMENT_R2 = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h5_harness_amendment_r2.json"
SCHEDULE = Path(
    os.environ.get(
        "QWEN35_H5_SCHEDULE",
        WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18_h5_schedule_materialization_attempt02_49876428/qwen35_c00_seed3407_010steps_080packs.json",
    )
)
HUMAN = Path(
    os.environ.get(
        "QWEN35_H5_HUMAN_PROTOCOL",
        WORKSPACE / "methodology/qwen35_hardware_qualification_r18_h5_protocol_r1_20260720.md",
    )
)
PREREGISTRATION = Path(
    os.environ.get(
        "QWEN35_H5_PREREGISTRATION_CLOSURE",
        WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/r18_h5_preregistration_closure_20260720.json",
    )
)
HARNESS_HUMAN_AMENDMENT = Path(
    os.environ.get(
        "QWEN35_H5_HARNESS_HUMAN_AMENDMENT",
        WORKSPACE / "methodology/qwen35_hardware_qualification_r18_h5_harness_amendment_r1_20260720.md",
    )
)
ATTEMPT01_FAILURE = Path(
    os.environ.get(
        "QWEN35_H5_ATTEMPT01_FAILURE_CLOSURE",
        WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/r18_h5_attempt01_failure_closure_20260720.json",
    )
)
HARNESS_HUMAN_AMENDMENT_R2 = Path(
    os.environ.get(
        "QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2",
        WORKSPACE / "methodology/qwen35_hardware_qualification_r18_h5_harness_amendment_r2_20260720.md",
    )
)
ATTEMPT02_FAILURE = Path(
    os.environ.get(
        "QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE",
        WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/r18_h5_attempt02_failure_closure_20260720.json",
    )
)
RELOAD_TYPE_DIAGNOSTIC = Path(
    os.environ.get(
        "QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC",
        WORKSPACE / "artifacts/qwen35_hardware_qualification_20260718/"
        "r18_h5_attempt02_reload_type_diagnostic_49881614/loading_info_type_report.json",
    )
)


def _contract() -> dict:
    value, digest = load_h5_contract(CONTRACT, human_protocol_path=HUMAN, preregistration_closure_path=PREREGISTRATION)
    assert digest == H5_CONTRACT_SHA256
    return value


def _harness_amendment() -> dict:
    value, digest = load_h5_harness_amendment(
        HARNESS_AMENDMENT,
        human_amendment_path=HARNESS_HUMAN_AMENDMENT,
        attempt01_failure_closure_path=ATTEMPT01_FAILURE,
    )
    assert digest == H5_HARNESS_AMENDMENT_SHA256
    return value


def _harness_amendment_r2() -> dict:
    value, digest = load_h5_harness_amendment_r2(
        HARNESS_AMENDMENT_R2,
        human_amendment_path=HARNESS_HUMAN_AMENDMENT_R2,
        attempt02_failure_closure_path=ATTEMPT02_FAILURE,
        reload_type_diagnostic_path=RELOAD_TYPE_DIAGNOSTIC,
    )
    assert digest == H5_HARNESS_AMENDMENT_R2_SHA256
    return value


def test_h5_harness_amendment_is_prospective_exact_and_non_scientific():
    value = _harness_amendment()
    assert value["failed_attempt"]["job_id"] == "49878043"
    assert value["full_h5_rerun_required"] is True
    assert value["scientific_training_authorized"] is False
    assert value["successor_on_complete_independent_pass"] == "H6_only"
    assert value["model_loss_data_schedule_optimizer_precision_or_threshold_change_allowed"] is False


def test_h5_harness_amendment_rejects_any_byte_mutation(tmp_path):
    changed = json.loads(HARNESS_AMENDMENT.read_text())
    changed["full_h5_rerun_required"] = False
    path = tmp_path / "changed-amendment.json"
    path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="harness-amendment digest drift"):
        load_h5_harness_amendment(
            path, human_amendment_path=HARNESS_HUMAN_AMENDMENT, attempt01_failure_closure_path=ATTEMPT01_FAILURE
        )


def test_h5_harness_amendment_r2_is_prospective_exact_and_non_scientific():
    value = _harness_amendment_r2()
    assert value["failed_attempt"]["job_id"] == "49880933"
    assert value["reload_type_diagnostic"]["set_paths"] == ["$.mismatched_keys", "$.missing_keys", "$.unexpected_keys"]
    assert value["full_h5_rerun_required"] is True
    assert value["scientific_training_authorized"] is False
    assert value["successor_on_complete_independent_pass"] == "H6_only"
    assert value["model_loss_data_schedule_optimizer_precision_or_threshold_change_allowed"] is False


def test_h5_harness_amendment_r2_rejects_any_byte_mutation(tmp_path):
    changed = json.loads(HARNESS_AMENDMENT_R2.read_text())
    changed["full_h5_rerun_required"] = False
    path = tmp_path / "changed-amendment-r2.json"
    path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="harness-amendment-R2 digest drift"):
        load_h5_harness_amendment_r2(
            path,
            human_amendment_path=HARNESS_HUMAN_AMENDMENT_R2,
            attempt02_failure_closure_path=ATTEMPT02_FAILURE,
            reload_type_diagnostic_path=RELOAD_TYPE_DIAGNOSTIC,
        )


def test_h5_r2_source_delta_is_exactly_bounded_from_preregistration_commit():
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = validate_h5_source_delta(
        ROOT,
        expected_head=head,
        harness_amendment_path=HARNESS_AMENDMENT,
        harness_human_amendment_path=HARNESS_HUMAN_AMENDMENT,
        attempt01_failure_closure_path=ATTEMPT01_FAILURE,
        harness_amendment_r2_path=HARNESS_AMENDMENT_R2,
        harness_human_amendment_r2_path=HARNESS_HUMAN_AMENDMENT_R2,
        attempt02_failure_closure_path=ATTEMPT02_FAILURE,
        reload_type_diagnostic_path=RELOAD_TYPE_DIAGNOSTIC,
    )
    assert result["status"] == "passed"
    assert result["preregistration_commit"] == "e20e4324cd837171b5a7d55626e81ecb54245d53"
    assert result["harness_amendment_r2_sha256"] == H5_HARNESS_AMENDMENT_R2_SHA256
    assert set(result["observed_changed_paths"]) <= set(result["allowed_paths"])


def test_h5_contract_is_exactly_hash_bound_and_scientifically_fail_closed():
    value = _contract()
    assert sha256_file(CONTRACT) == H5_CONTRACT_SHA256
    assert sha256_file(HUMAN) == H5_HUMAN_PROTOCOL_SHA256
    assert sha256_file(PREREGISTRATION) == H5_PREREGISTRATION_CLOSURE_SHA256
    assert value["execution"]["selected_loss_chunk_size"] == H5_SELECTED_CHUNK_SIZE == 512
    assert value["scientific_training_authorized"] is False
    assert value["automatic_successor"] is False
    assert value["allowed_successor_on_complete_pass"] == "H6_only"


def test_h5_contract_rejects_byte_or_semantic_drift(tmp_path):
    changed = json.loads(CONTRACT.read_text())
    changed["execution"]["selected_loss_chunk_size"] = 256
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="contract digest drift"):
        load_h5_contract(path, human_protocol_path=HUMAN, preregistration_closure_path=PREREGISTRATION)


def test_h5_schedule_exact_prefix_and_accounting_are_independently_recomputed():
    contract = _contract()
    schedule, prefix = _validate_schedule(SimpleNamespace(schedule=SCHEDULE), contract)
    assert schedule["schedule_sha256"] == contract["schedule"]["embedded_schedule_sha256"]
    assert len(prefix) == 40
    assert [row["schedule_index"] for row in prefix] == list(range(40))
    assert len({row["pack_uid"] for row in prefix}) == 40
    assert len({row["pack_index"] for row in prefix}) == 40
    assert contract["five_update_exposure"]["entries_sha256"] == H5_FIRST_FIVE_ENTRIES_SHA256
    observed_targets = [
        sum(row["assistant_targets"] for row in prefix[step * 8 : (step + 1) * 8]) for step in range(5)
    ]
    assert observed_targets == list(H5_EXPECTED_TARGETS_BY_UPDATE)
    assert sum(observed_targets) == EXPECTED_EXPOSURE_TOTALS["assistant_targets"]


def test_h5_schedule_validator_rejects_any_file_mutation_before_semantic_use(tmp_path):
    value = json.loads(SCHEDULE.read_text())
    mutations = []
    duplicate = copy.deepcopy(value)
    duplicate["entries"][1]["pack_uid"] = duplicate["entries"][0]["pack_uid"]
    mutations.append(duplicate)
    wrong_index = copy.deepcopy(value)
    wrong_index["entries"][3]["schedule_index"] = 2
    mutations.append(wrong_index)
    wrong_target = copy.deepcopy(value)
    wrong_target["entries"][0]["assistant_targets"] += 1
    mutations.append(wrong_target)
    for index, mutation in enumerate(mutations):
        path = tmp_path / f"mutation-{index}.json"
        path.write_text(json.dumps(mutation, sort_keys=True) + "\n")
        with pytest.raises(ValueError, match="schedule file digest drift"):
            _validate_schedule(SimpleNamespace(schedule=path), _contract())


def _trace(*events: dict) -> dict:
    return {"traceEvents": list(events)}


def test_h5_nccl_trace_catalog_accepts_exact_positive_complete_events(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text("{}")
    result = validate_trace(
        _trace(
            {"cat": "kernel", "name": "ncclDevKernel_AllReduce_RING_LL", "ph": "X", "dur": 11.5},
            {"cat": "cpu_op", "name": "c10d::allreduce_", "ph": "X", "dur": 2.0},
            {"cat": "cpu_op", "name": "aten::linear", "ph": "X", "dur": 4.0},
        ),
        trace_path=path,
    )
    assert result["distinct_collective_event_names"] == 2
    assert result["all_reduce_event_name_count"] == 2
    assert result["collective_complete_event_count"] == 2
    assert result["collective_complete_event_duration_microseconds_sum_with_overlap"] == 13.5


@pytest.mark.parametrize("chunk_size", range(1, len(b'"Process Group Description": ,') + 2))
def test_h5_trace_sanitizer_is_boundary_safe_and_changes_only_exact_field(tmp_path, chunk_size):
    raw = tmp_path / "raw.json"
    output = tmp_path / "sanitized.json"
    raw_bytes = b'{"traceEvents":[{"name":"nccl allreduce","args":{"Process Group Description": ,"duration":1}}]}'
    raw.write_bytes(raw_bytes)
    report = sanitize_profiler_trace(raw, output, chunk_size=chunk_size)
    expected = raw_bytes.replace(b'"Process Group Description": ,', b'"Process Group Description": null,')
    assert output.read_bytes() == expected
    assert report["replacement_count"] == 1
    assert report["sanitized_trace_bytes"] == report["raw_trace_bytes"] + 4
    assert json.loads(output.read_text())["traceEvents"][0]["args"]["Process Group Description"] is None


@pytest.mark.parametrize("split_after_bytes", range(1, len(b'"Process Group Description": ,')))
def test_h5_trace_sanitizer_handles_every_internal_pattern_split(tmp_path, split_after_bytes):
    pattern = b'"Process Group Description": ,'
    chunk_size = 64
    prefix = b"x" * (chunk_size - split_after_bytes)
    raw = tmp_path / "raw.bin"
    output = tmp_path / "sanitized.bin"
    raw_bytes = prefix + pattern + b"tail"
    raw.write_bytes(raw_bytes)
    report = sanitize_profiler_trace(raw, output, chunk_size=chunk_size)
    assert output.read_bytes() == raw_bytes.replace(pattern, b'"Process Group Description": null,')
    assert report["replacement_count"] == 1


def test_h5_trace_sanitizer_repairs_multiple_fields_and_is_identity_on_valid_trace(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed_output = tmp_path / "malformed-sanitized.json"
    malformed.write_bytes(
        b'{"a":{"Process Group Description": },"b":{"Process Group Description": }}'.replace(b'": }', b'": ,"x":1}')
    )
    report = sanitize_profiler_trace(malformed, malformed_output, chunk_size=7)
    assert report["replacement_count"] == 2
    json.loads(malformed_output.read_text())

    valid = tmp_path / "valid.json"
    valid_output = tmp_path / "valid-sanitized.json"
    valid.write_text('{"traceEvents":[]}\n')
    identity = sanitize_profiler_trace(valid, valid_output, chunk_size=3)
    assert identity["replacement_count"] == 0
    assert valid.read_bytes() == valid_output.read_bytes()
    assert identity["raw_trace_sha256"] == identity["sanitized_trace_sha256"]


def test_h5_trace_sanitizer_rejects_unrecognized_empty_value_and_existing_output(tmp_path):
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_bytes(b'{"unrelated": ,"traceEvents":[]}')
    output = tmp_path / "sanitized.json"
    with pytest.raises(AssertionError, match="unrecognized empty JSON value"):
        sanitize_profiler_trace(unrelated, output, chunk_size=2)
    assert not output.exists()

    valid = tmp_path / "valid.json"
    valid.write_text('{"traceEvents":[]}')
    output.write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        sanitize_profiler_trace(valid, output)


@pytest.mark.parametrize(
    ("raw_bytes", "message"),
    [
        (
            b'{"traceEvents":[],"duplicate":1,"duplicate":2,"Process Group Description": }'.replace(
                b'": }', b'": ,"x":1}'
            ),
            "duplicate JSON object key",
        ),
        (
            b'{"traceEvents":[],"nonfinite":Infinity,"Process Group Description": }'.replace(b'": }', b'": ,"x":1}'),
            "non-finite",
        ),
    ],
)
def test_h5_trace_sanitizer_does_not_weaken_strict_json_policy(tmp_path, raw_bytes, message):
    raw = tmp_path / "raw.json"
    output = tmp_path / "sanitized.json"
    raw.write_bytes(raw_bytes)
    sanitize_profiler_trace(raw, output, chunk_size=5)
    with pytest.raises(ValueError, match=message):
        load_strict_json(output)


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        (_trace({"cat": "cpu", "name": "aten::linear", "ph": "X", "dur": 1}), "no NCCL/collective"),
        (_trace({"cat": "cpu", "name": "c10d::broadcast", "ph": "X", "dur": 1}), "no NCCL-identified"),
        (_trace({"cat": "gpu", "name": "nccl broadcast", "ph": "X", "dur": 1}), "no all-reduce"),
        (_trace({"cat": "gpu", "name": "nccl allreduce timeout", "ph": "X", "dur": 1}), "error-like"),
        (_trace({"cat": "gpu", "name": "nccl allreduce", "ph": "X", "dur": 0}), "no positive-duration"),
        (_trace({"cat": "gpu", "name": "nccl allreduce", "ph": "X", "dur": math.nan}), "invalid duration"),
    ],
)
def test_h5_nccl_trace_catalog_rejects_missing_or_invalid_collective_evidence(tmp_path, trace, message):
    path = tmp_path / "trace.json"
    path.write_text("{}")
    with pytest.raises((ValueError, AssertionError), match=message):
        validate_trace(trace, trace_path=path)


def test_h5_learning_rate_trajectory_uses_ten_step_horizon_despite_step_five_stop():
    values = _expected_learning_rates()
    assert len(values) == 5
    assert values[0] == 0.0
    assert values[1] == pytest.approx(2e-5)
    assert values[2] == pytest.approx(2e-5 * 0.5 * (1 + math.cos(math.pi / 9)))
    assert values[-1] > 0


def test_h5_wrapper_is_bounded_exact_and_has_no_automatic_successor():
    source = WRAPPER.read_text()
    directives = dict(re.findall(r"^#SBATCH --([^=]+)=(.+)$", source, flags=re.MULTILINE))
    assert directives["account"] == "aifac_f02_434"
    assert directives["nodes"] == "1"
    assert directives["gres"] == "gpu:4"
    assert directives["time"] == "00:45:00"
    assert source.count("--nproc_per_node=4") == 3
    for exact in (
        "--selected_loss_chunk_size 512",
        "--gradient_accumulation_steps 2",
        "--max_steps 10",
        "--stop_after_steps 5",
        "--expected_final_global_step 5",
        "--hardware_profile true",
        "--cuda_event_step_timing true",
        "--average_tokens_across_devices true",
        "--require_no_dense_logits true",
        "--require_forward_loss_audit true",
    ):
        assert exact in source
    assert "use_liger_fused_linear_cross_entropy false" in source
    assert "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in source
    assert "export NCCL_ASYNC_ERROR_HANDLING" not in source
    assert "export TORCH_DISTRIBUTED_DEBUG=INFO" in source
    assert "TORCH_DISTRIBUTED_DEBUG=DETAIL" not in source
    assert "--sanitized-trace-output" in source
    for binding in (
        "QWEN35_H5_HARNESS_AMENDMENT_R2",
        "QWEN35_H5_HARNESS_HUMAN_AMENDMENT_R2",
        "QWEN35_H5_ATTEMPT02_FAILURE_CLOSURE",
        "QWEN35_H5_RELOAD_TYPE_DIAGNOSTIC",
        "--harness-amendment-r2",
        "--harness-human-amendment-r2",
        "--attempt02-failure-closure",
        "--reload-type-diagnostic",
    ):
        assert binding in source
    assert H5_HARNESS_AMENDMENT_R2_SHA256 in source
    assert "sbatch " not in source
    assert "leonardo_h6" not in source
    assert "C01" not in source and "C11" not in source


def test_h5_every_distributed_phase_has_explicit_device_or_shutdown_controls():
    trainer = (ROOT / "scripts/train/qwen35/train_qwen35_sft.py").read_text()
    sharding = (ROOT / "scripts/train/qwen35/validate_qwen35_accelerate_schedule_sharding_r18_h5.py").read_text()
    ddp = (ROOT / "scripts/train/qwen35/validate_qwen35_ddp_loss_normalization_r18_h5.py").read_text()
    trainer_tree = ast.parse(trainer)
    trainer_barriers = []
    for node in ast.walk(trainer_tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "barrier":
            continue
        distributed = node.func.value
        if (
            isinstance(distributed, ast.Attribute)
            and distributed.attr == "distributed"
            and isinstance(distributed.value, ast.Name)
            and distributed.value.id == "torch"
        ):
            trainer_barriers.append(node)
    assert len(trainer_barriers) == 1
    assert all(any(keyword.arg == "device_ids" for keyword in call.keywords) for call in trainer_barriers)
    assert "distributed_barrier_on_local_cuda_device()" in trainer
    assert "finally:" in trainer and "destroy_initialized_process_group()" in trainer
    assert "Do not put a barrier in this exceptional path" in trainer
    assert "accelerator.end_training()" in sharding
    assert "torch.distributed.destroy_process_group()" in sharding
    assert 'init_process_group("nccl", device_id=device)' in ddp
    assert "barrier(device_ids=[local_rank])" in ddp
    assert "torch.distributed.destroy_process_group()" in ddp


@pytest.mark.parametrize("container_type", [list, tuple, set, frozenset])
def test_h5_loading_info_normalization_accepts_only_frozen_container_types(container_type):
    values = container_type(["z", "a"])
    result = normalize_loading_info(
        {
            "missing_keys": values,
            "unexpected_keys": container_type(),
            "mismatched_keys": container_type(),
            "error_msgs": container_type(),
        }
    )
    assert result == {"missing_keys": ["a", "z"], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    json.dumps(result, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("loading_info", "error", "message"),
    [
        (
            {"missing_keys": ["x", "x"], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
            ValueError,
            "duplicate",
        ),
        (
            {"missing_keys": [1], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
            TypeError,
            "non-string",
        ),
        (
            {"missing_keys": {"x": 1}, "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
            TypeError,
            "unsupported container",
        ),
        (
            {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []},
            TypeError,
            "exactly the four pinned keys",
        ),
        (["not", "a", "mapping"], TypeError, "mapping"),
    ],
)
def test_h5_loading_info_normalization_rejects_ambiguous_or_lossy_inputs(loading_info, error, message):
    with pytest.raises(error, match=message):
        normalize_loading_info(loading_info)


@pytest.mark.parametrize(
    "marker",
    [
        "destroy_process_group() was not called before program exit",
        "barrier(): using the device under current context",
        "Guessing device ID based on global rank",
        "Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated",
        "NCCL WARN collective failure",
        "collective timeout",
    ],
)
def test_h5_process_failure_scanner_rejects_lifecycle_and_nccl_markers(marker):
    assert NCCL_FAILURE_PATTERN.search(marker)


def test_h5_preflights_use_the_actual_non_liger_and_accelerate_primitives():
    sharding = (ROOT / "scripts/train/qwen35/validate_qwen35_accelerate_schedule_sharding_r18_h5.py").read_text()
    ddp = (ROOT / "scripts/train/qwen35/validate_qwen35_ddp_loss_normalization_r18_h5.py").read_text()
    assert "DataLoaderConfiguration(even_batches=False)" in sharding
    assert "list(range(accelerator.process_index, 40, H5_WORLD_SIZE))" in sharding
    assert "checkpointed_chunked_selective_linear_cross_entropy" in ddp
    assert "ordinary_chunked_selective_linear_cross_entropy" in ddp
    assert "TARGET_COUNTS = (0, 127, 513, 1025)" in ddp
    assert "return loss * world_size" in ddp
    assert "from liger" not in ddp and "import liger" not in ddp


def test_h5_strict_json_rejects_duplicate_keys_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":Infinity}\n')
    with pytest.raises(ValueError, match="non-finite"):
        load_strict_json(nonfinite)
