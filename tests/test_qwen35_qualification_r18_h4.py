from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.train.qwen35.capture_qwen35_h4_job_identity_r18 import _nvidia_inventory
from scripts.train.qwen35.finalize_qwen35_h4_kernel_audit_r18 import validate as validate_kernel_audit
from scripts.train.qwen35.train_qwen35_sft import DataArguments, ModelArguments, Qwen35TrainingArguments
from scripts.train.qwen35.validate_qwen35_h4_candidate_r18 import CUDA_TIMING_SCOPE, _validate_cuda_timing_artifact
from scripts.train.qwen35.validate_qwen35_h4_set_r18 import validate as validate_h4_set

from open_instruct.qwen35_chunked_loss import IMPLEMENTATION_ID
from open_instruct.qwen35_hardware import Qwen35CudaEventTimerCallback, Qwen35HardwareProfilerCallback
from open_instruct.qwen35_qualification_r18_h4 import (
    H4_ALLOCATOR_HISTORY_ENTRY_CAP,
    H4_CONTRACT_SHA256,
    LEONARDO_A100_COMPUTE_CAPABILITY,
    LEONARDO_A100_MEMORY_MIB,
    LEONARDO_A100_NAME,
    inventory_chrome_trace,
    load_h4_contract,
    load_strict_json,
    select_chunk_size,
    sha256_file,
    timing_statistics,
    validate_forward_loss_audit,
    validate_memory_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h4.json"
H4_WRAPPER = ROOT / "scripts/train/qwen35/leonardo_h4_candidate_r18.sbatch"


def valid_audit(*, rows: int = 257, chunk_size: int = 128, divisor: int = 999) -> dict:
    boundaries = [[start, min(start + chunk_size, rows)] for start in range(0, rows, chunk_size)]
    maximum = max((end - start for start, end in boundaries), default=0)
    return {
        "checkpointed": True,
        "chunk_boundaries": boundaries,
        "chunk_count": len(boundaries),
        "chunk_size": chunk_size,
        "full_selected_logit_elements": rows * 248320,
        "global_target_count": divisor,
        "hidden_size": 1024,
        "implementation_id": IMPLEMENTATION_ID,
        "maximum_chunk_rows": maximum,
        "maximum_logit_elements": maximum * 248320,
        "returned_dense_logits": False,
        "selected_rows": rows,
        "vocabulary_size": 248320,
        "zero_target": rows == 0,
    }


def test_h4_contract_is_exactly_hash_bound_and_preregistered():
    value, digest = load_h4_contract(CONTRACT)
    assert digest == H4_CONTRACT_SHA256 == sha256_file(CONTRACT)
    assert value["candidate_chunk_sizes_in_execution_order"] == [128, 256, 512, 1024]
    assert value["scientific_training_authorized"] is False
    assert value["allowed_successor_on_complete_pass"] == "H5_only"
    assert LEONARDO_A100_NAME == "NVIDIA A100-SXM-64GB"
    assert LEONARDO_A100_MEMORY_MIB == "65536"
    assert LEONARDO_A100_COMPUTE_CAPABILITY == "8.0"


def test_every_h4_gpu_identity_consumer_imports_the_single_canonical_identity():
    consumers = [
        ROOT / "scripts/train/qwen35/capture_qwen35_h4_job_identity_r18.py",
        ROOT / "scripts/train/qwen35/validate_qwen35_h4_profile_r18.py",
        ROOT / "scripts/train/qwen35/validate_qwen35_h4_candidate_r18.py",
    ]
    for path in consumers:
        source = path.read_text()
        assert "LEONARDO_A100_NAME" in source
        assert "NVIDIA A100-SXM4-64GB" not in source


def test_every_h4_allocator_history_consumer_imports_the_single_canonical_cap():
    callback = ROOT / "open_instruct/qwen35_hardware.py"
    validator = ROOT / "scripts/train/qwen35/validate_qwen35_h4_profile_r18.py"
    assert H4_ALLOCATOR_HISTORY_ENTRY_CAP == 2_000_000
    assert Qwen35HardwareProfilerCallback.HISTORY_ENTRY_CAP == H4_ALLOCATOR_HISTORY_ENTRY_CAP
    for path in (callback, validator):
        source = path.read_text()
        assert "H4_ALLOCATOR_HISTORY_ENTRY_CAP" in source
        assert '"maximum_entries": 100000' not in source
        assert '"maximum_entries": 2_000_000' not in source


def test_cuda_event_timer_uses_training_arguments_process_identity(monkeypatch, tmp_path):
    class FakeStartEvent:
        def __init__(self, step):
            self.step = step

        def elapsed_time(self, end):
            assert end.step == self.step
            return float(self.step) + 0.25

    class FakeEndEvent:
        def __init__(self, step):
            self.step = step

    callback = object.__new__(Qwen35CudaEventTimerCallback)
    callback.expected_steps = 13
    callback.starts = {step: FakeStartEvent(step) for step in range(1, 14)}
    callback.ends = {step: FakeEndEvent(step) for step in range(1, 14)}
    callback.output_dir = tmp_path
    callback.manifest = {"protocol_id": "qualification"}
    callback.manifest_sha256 = "qualification-sha256"
    callback.h4 = {"protocol_id": "h4"}
    callback.h4_sha256 = "h4-sha256"
    callback.candidate_chunk_size = 128
    synchronize_calls = []
    monkeypatch.setattr("open_instruct.qwen35_hardware.torch.cuda.synchronize", lambda: synchronize_calls.append(True))
    args = SimpleNamespace(process_index=7, world_size=8)
    state = SimpleNamespace(global_step=13)
    control = object()

    assert callback.on_train_end(args, state, control) is control
    assert synchronize_calls == [True]
    report = json.loads((tmp_path / "qwen35_cuda_step_times_rank07.json").read_text())
    assert report == {
        "artifact": "qwen35_per_rank_cuda_event_step_timing",
        "assay": "timing",
        "candidate_chunk_size": 128,
        "completed_optimizer_steps": 13,
        "cuda_event_step_milliseconds": {str(step): float(step) + 0.25 for step in range(1, 14)},
        "h4_contract_sha256": "h4-sha256",
        "h4_protocol_id": "h4",
        "qualification_manifest_sha256": "qualification-sha256",
        "qualification_protocol_id": "qualification",
        "rank": 7,
        "schema_version": 1,
        "status": "passed",
        "timing_scope": (
            "rank-local default-stream events around Trainer optimizer step; synchronized once at train end"
        ),
        "world_size": 8,
    }


def _valid_cuda_timing_artifact() -> dict:
    return {
        "artifact": "qwen35_per_rank_cuda_event_step_timing",
        "assay": "timing",
        "candidate_chunk_size": 128,
        "completed_optimizer_steps": 13,
        "cuda_event_step_milliseconds": {str(step): float(step) + 0.25 for step in range(1, 14)},
        "h4_contract_sha256": "h4-sha256",
        "h4_protocol_id": "h4",
        "qualification_manifest_sha256": "qualification-sha256",
        "qualification_protocol_id": "qualification",
        "rank": 0,
        "schema_version": 1,
        "status": "passed",
        "timing_scope": CUDA_TIMING_SCOPE,
        "world_size": 1,
    }


def _validate_test_cuda_timing(value: dict) -> dict[str, float]:
    return _validate_cuda_timing_artifact(
        value,
        candidate_chunk_size=128,
        qualification={"protocol_id": "qualification"},
        qualification_sha256="qualification-sha256",
        h4={"protocol_id": "h4"},
        h4_sha256="h4-sha256",
    )


def test_h4_cuda_timing_validator_accepts_canonical_sorted_json_object_order(tmp_path):
    path = tmp_path / "timing.json"
    _write_json(path, _valid_cuda_timing_artifact())
    loaded = load_strict_json(path)
    assert list(loaded["cuda_event_step_milliseconds"]) == [
        "1",
        "10",
        "11",
        "12",
        "13",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    assert _validate_test_cuda_timing(loaded) == {str(step): float(step) + 0.25 for step in range(1, 14)}


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("artifact", "wrong"),
        ("schema_version", 2),
        ("assay", "profiler"),
        ("candidate_chunk_size", 256),
        ("completed_optimizer_steps", 12),
        ("h4_contract_sha256", "wrong"),
        ("h4_protocol_id", "wrong"),
        ("qualification_manifest_sha256", "wrong"),
        ("qualification_protocol_id", "wrong"),
        ("rank", 1),
        ("world_size", 2),
        ("status", "failed"),
        ("timing_scope", "wrong"),
    ],
)
def test_h4_cuda_timing_validator_rejects_identity_or_scope_drift(field, bad_value):
    value = _valid_cuda_timing_artifact()
    value[field] = bad_value
    with pytest.raises(ValueError, match="artifact drift"):
        _validate_test_cuda_timing(value)


def test_h4_cuda_timing_validator_rejects_step_set_duration_or_field_drift():
    missing = _valid_cuda_timing_artifact()
    missing["cuda_event_step_milliseconds"].pop("13")
    with pytest.raises(ValueError, match="step set drift"):
        _validate_test_cuda_timing(missing)
    additional = _valid_cuda_timing_artifact()
    additional["cuda_event_step_milliseconds"]["14"] = 1.0
    with pytest.raises(ValueError, match="step set drift"):
        _validate_test_cuda_timing(additional)
    leading_zero = _valid_cuda_timing_artifact()
    leading_zero["cuda_event_step_milliseconds"]["01"] = leading_zero["cuda_event_step_milliseconds"].pop("1")
    with pytest.raises(ValueError, match="step set drift"):
        _validate_test_cuda_timing(leading_zero)
    for duration, message in (
        (0.0, "nonpositive"),
        (math.inf, "nonpositive"),
        (True, "not numeric"),
        ("1", "not numeric"),
    ):
        invalid = _valid_cuda_timing_artifact()
        invalid["cuda_event_step_milliseconds"]["1"] = duration
        with pytest.raises(ValueError, match=message):
            _validate_test_cuda_timing(invalid)
    extra_field = _valid_cuda_timing_artifact()
    extra_field["unregistered"] = 1
    with pytest.raises(ValueError, match="field drift"):
        _validate_test_cuda_timing(extra_field)


def test_h4_contract_rejects_byte_or_semantic_drift(tmp_path):
    value = json.loads(CONTRACT.read_text())
    value["timing_selection"]["tie_fraction_inclusive"] = 0.021
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="contract digest drift"):
        load_h4_contract(changed)


def test_h4_gpu_inventory_accepts_only_exact_recorded_leonardo_a100_identity(monkeypatch):
    observed = "0, NVIDIA A100-SXM-64GB, GPU-00000000-0000-0000-0000-000000000000, 65536, 535.274.02, 8.0\n"
    monkeypatch.setattr(
        "scripts.train.qwen35.capture_qwen35_h4_job_identity_r18.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=observed),
    )
    rows = _nvidia_inventory()
    assert rows == [
        {
            "compute_cap": "8.0",
            "driver_version": "535.274.02",
            "index": "0",
            "memory.total": "65536",
            "name": "NVIDIA A100-SXM-64GB",
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
        }
    ]

    for wrong in (
        observed.replace("NVIDIA A100-SXM-64GB", "NVIDIA A100-SXM4-64GB"),
        observed.replace("65536", "40960"),
        observed.replace("8.0", "9.0"),
        observed + observed.replace("0,", "1,", 1),
    ):
        monkeypatch.setattr(
            "scripts.train.qwen35.capture_qwen35_h4_job_identity_r18.subprocess.run",
            lambda *args, _wrong=wrong, **kwargs: SimpleNamespace(stdout=_wrong),
        )
        with pytest.raises(ValueError, match="H4 .* visible GPU"):
            _nvidia_inventory()


def test_h4_wrapper_train_invocations_use_only_current_pinned_parser_fields():
    wrapper = H4_WRAPPER.read_text()
    common = wrapper.split("common_arguments=(", 1)[1].split("\n)", 1)[0]
    train_invocations = re.findall(
        r'"\$QWEN35_VENV/bin/python" scripts/train/qwen35/train_qwen35_sft\.py \\\n(.*?)(?=\n\n)',
        wrapper,
        flags=re.DOTALL,
    )
    assert len(train_invocations) == 2
    observed_flags = set(re.findall(r"--([a-zA-Z0-9_]+)", common))
    for invocation in train_invocations:
        observed_flags.update(re.findall(r"--([a-zA-Z0-9_]+)", invocation))
    parser_fields = {
        field.name
        for argument_type in (ModelArguments, DataArguments, Qwen35TrainingArguments)
        for field in dataclasses.fields(argument_type)
    }
    assert observed_flags <= parser_fields, sorted(observed_flags - parser_fields)
    assert "save_safetensors" not in observed_flags


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n')
    with pytest.raises(ValueError, match="non-finite"):
        load_strict_json(nonfinite)


def test_forward_loss_audit_recomputes_every_derived_field():
    audit = valid_audit()
    result = validate_forward_loss_audit(
        audit, expected_selected_rows=257, expected_global_target_count=999, expected_chunk_size=128
    )
    assert result == {
        "chunk_count": 3,
        "full_selected_logit_elements": 257 * 248320,
        "global_target_count": 999,
        "maximum_chunk_rows": 128,
        "maximum_logit_elements": 128 * 248320,
        "selected_rows": 257,
        "status": "passed",
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("implementation_id", "wrong"),
        ("checkpointed", False),
        ("selected_rows", 256),
        ("chunk_size", 256),
        ("chunk_count", 2),
        ("chunk_boundaries", [[0, 128], [129, 257]]),
        ("maximum_chunk_rows", 129),
        ("vocabulary_size", 248319),
        ("hidden_size", 1023),
        ("maximum_logit_elements", 1),
        ("full_selected_logit_elements", 1),
        ("global_target_count", 998),
        ("zero_target", True),
        ("returned_dense_logits", True),
    ],
)
def test_forward_loss_audit_rejects_every_field_drift(field, bad_value):
    audit = valid_audit()
    audit[field] = bad_value
    with pytest.raises(ValueError):
        validate_forward_loss_audit(
            audit, expected_selected_rows=257, expected_global_target_count=999, expected_chunk_size=128
        )


def test_forward_loss_audit_rejects_missing_extra_or_nonfinite_fields():
    missing = valid_audit()
    missing.pop("chunk_count")
    with pytest.raises(ValueError, match="field drift"):
        validate_forward_loss_audit(
            missing, expected_selected_rows=257, expected_global_target_count=999, expected_chunk_size=128
        )
    extra = valid_audit()
    extra["unregistered"] = 1
    with pytest.raises(ValueError, match="field drift"):
        validate_forward_loss_audit(
            extra, expected_selected_rows=257, expected_global_target_count=999, expected_chunk_size=128
        )
    nonfinite = valid_audit()
    nonfinite["maximum_logit_elements"] = math.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_forward_loss_audit(
            nonfinite, expected_selected_rows=257, expected_global_target_count=999, expected_chunk_size=128
        )


def test_trace_inventory_uses_exact_names_counts_categories_and_durations():
    trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "exact_kernel_A", "dur": 1.25},
            {"cat": "kernel", "name": "exact_kernel_A", "dur": 2.75},
            {"cat": "gpu_memcpy", "name": "Memcpy DtoD", "dur": 3},
            {"cat": "cpu_op", "name": "aten::linear", "dur": 10, "args": {"shape": [2, 3]}},
        ]
    }
    result = inventory_chrome_trace(trace)
    assert result["trace_event_count"] == 4
    assert result["accelerator_event_count"] == 3
    assert result["distinct_accelerator_event_names"] == 2
    assert result["observed_accelerator_events"] == [
        {"categories": ["gpu_memcpy"], "count": 1, "duration_microseconds": 3.0, "exact_name": "Memcpy DtoD"},
        {"categories": ["kernel"], "count": 2, "duration_microseconds": 4.0, "exact_name": "exact_kernel_A"},
    ]


@pytest.mark.parametrize("shape", ([1, 32768, 248320], [32768, 248320], "shape=[1, 32768, 248320]", "[32768,248320]"))
def test_trace_inventory_rejects_dense_full_sequence_logits(shape):
    trace = {"traceEvents": [{"cat": "kernel", "name": "kernel", "dur": 1, "args": {"shape": shape}}]}
    with pytest.raises(AssertionError, match="dense full-sequence"):
        inventory_chrome_trace(trace)


def test_trace_inventory_rejects_missing_kernel_name_bad_duration_or_no_accelerator_events():
    with pytest.raises(ValueError, match="no exact name"):
        inventory_chrome_trace({"traceEvents": [{"cat": "kernel", "dur": 1}]})
    with pytest.raises(ValueError, match="invalid duration"):
        inventory_chrome_trace({"traceEvents": [{"cat": "kernel", "name": "k", "dur": -1}]})
    with pytest.raises(ValueError, match="no accelerator"):
        inventory_chrome_trace({"traceEvents": [{"cat": "cpu_op", "name": "aten::mm", "dur": 1}]})


def test_memory_snapshot_validates_schema_actions_and_history_headroom():
    snapshot = {
        "segments": [{"device": 0}],
        "device_traces": [[{"action": "alloc"}, {"action": "free_requested"}, {"action": "free_completed"}]],
        "allocator_settings": "expandable_segments:False",
    }
    result = validate_memory_snapshot(snapshot, history_entry_cap=4)
    assert result["action_counts"] == {"alloc": 1, "free_completed": 1, "free_requested": 1}
    assert result["device_trace_lengths"] == [3]
    assert result["history_entry_cap_reached"] is False


def test_memory_snapshot_rejects_cap_reach_oom_or_schema_drift():
    with pytest.raises(AssertionError, match="reached"):
        validate_memory_snapshot(
            {"segments": [], "device_traces": [[{"action": "alloc"}, {"action": "free"}]]}, history_entry_cap=2
        )
    with pytest.raises(AssertionError, match="OOM"):
        validate_memory_snapshot({"segments": [], "device_traces": [[{"action": "oom"}]]}, history_entry_cap=2)
    with pytest.raises(ValueError, match="lacks segments"):
        validate_memory_snapshot({"segments": [], "device_traces": []}, history_entry_cap=2)


def test_real_h4_allocator_history_cap_accepts_cap_minus_one_and_rejects_cap():
    cap = H4_ALLOCATOR_HISTORY_ENTRY_CAP
    assert cap == Qwen35HardwareProfilerCallback.HISTORY_ENTRY_CAP
    event = {"action": "alloc"}
    below_cap = validate_memory_snapshot(
        {"segments": [], "device_traces": [[event] * (cap - 1)]}, history_entry_cap=cap
    )
    assert below_cap["device_trace_lengths"] == [cap - 1]
    assert below_cap["action_counts"] == {"alloc": cap - 1}
    with pytest.raises(AssertionError, match="reached"):
        validate_memory_snapshot({"segments": [], "device_traces": [[event] * cap]}, history_entry_cap=cap)


def test_timing_statistics_use_sample_standard_deviation_and_exact_cardinality():
    values = [float(value) for value in range(1, 11)]
    result = timing_statistics(values)
    assert result["count"] == 10
    assert result["median_seconds"] == 5.5
    assert result["mean_seconds"] == 5.5
    assert result["sample_standard_deviation_seconds"] == pytest.approx(3.0276503540974917)
    assert result["coefficient_of_variation"] == pytest.approx(3.0276503540974917 / 5.5)
    with pytest.raises(ValueError, match="exactly ten"):
        timing_statistics(values[:-1])
    with pytest.raises(ValueError, match="finite and positive"):
        timing_statistics([1.0] * 9 + [0.0])


def test_chunk_selection_uses_inclusive_two_percent_tie_and_smallest_chunk():
    contract, _ = load_h4_contract(CONTRACT)
    medians = {128: 10.2, 256: 10.0, 512: 10.201, 1024: 10.5}
    rows = []
    for chunk_size in contract["candidate_chunk_sizes_in_execution_order"]:
        values = [medians[chunk_size]] * 10
        rows.append(
            {
                "chunk_size": chunk_size,
                "eligible": True,
                "measured_update_seconds": values,
                "timing_statistics": timing_statistics(values),
            }
        )
    result = select_chunk_size(rows, contract)
    assert result["tied_chunk_sizes"] == [128, 256]
    assert result["selected_chunk_size"] == 128


def test_chunk_selection_rejects_candidate_drift_ineligible_or_unstable_rows():
    contract, _ = load_h4_contract(CONTRACT)
    rows = [
        {
            "chunk_size": chunk_size,
            "eligible": True,
            "measured_update_seconds": [10.0] * 10,
            "timing_statistics": timing_statistics([10.0] * 10),
        }
        for chunk_size in contract["candidate_chunk_sizes_in_execution_order"]
    ]
    with pytest.raises(ValueError, match="candidate order"):
        select_chunk_size(list(reversed(rows)), contract)
    ineligible = copy.deepcopy(rows)
    ineligible[0]["eligible"] = False
    with pytest.raises(AssertionError, match="not memory/kernel/update eligible"):
        select_chunk_size(ineligible, contract)
    unstable = copy.deepcopy(rows)
    unstable[0]["measured_update_seconds"] = [1.0, 20.0] * 5
    unstable[0]["timing_statistics"] = timing_statistics(unstable[0]["measured_update_seconds"])
    with pytest.raises(AssertionError, match="CV exceeds"):
        select_chunk_size(unstable, contract)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def test_kernel_finalizer_requires_exact_observed_set_and_all_component_evidence(tmp_path):
    contract, contract_sha256 = load_h4_contract(CONTRACT)
    profile_path = tmp_path / "profile.json"
    mapping_path = tmp_path / "mapping.json"
    observed = {
        "categories": ["kernel"],
        "count": 7,
        "duration_microseconds": 12.5,
        "exact_name": "exact_generated_kernel_name",
    }
    _write_json(
        profile_path,
        {
            "candidate_chunk_size": 128,
            "h4_contract_sha256": contract_sha256,
            "status": "automated_profile_passed_pending_manual_kernel_mapping",
            "trace_inventory": {
                "accelerator_event_count": 7,
                "observed_accelerator_events": [observed],
                "observed_all_event_names": [
                    {"categories": ["kernel"], "count": 7, "exact_name": observed["exact_name"]}
                ],
            },
            "trace_sha256": "a" * 64,
        },
    )
    components = contract["kernel_path"]["required_components"]
    _write_json(
        mapping_path,
        {
            "accelerator_events": [
                {
                    "disposition": "allowed",
                    "exact_name": observed["exact_name"],
                    "observed_categories": observed["categories"],
                    "observed_count": observed["count"],
                    "observed_duration_microseconds": observed["duration_microseconds"],
                    "rationale": "Test-only exact mapping row.",
                    "semantic_components": components,
                    "source_file_or_implementation_family": "test/source.py",
                    "source_identity": "test-package-1.0",
                }
            ],
            "artifact": "qwen35_r18_h4_reviewed_kernel_mapping",
            "candidate_chunk_size": 128,
            "h4_contract_sha256": contract_sha256,
            "liger_execution_observed": False,
            "required_component_evidence": {
                component: [
                    {
                        "observed_exact_event_name": observed["exact_name"],
                        "rationale": "Test-only exact component evidence.",
                        "source_file_or_implementation_family": "test/source.py",
                        "source_identity": "test-package-1.0",
                    }
                ]
                for component in components
            },
            "review_status": "reviewed_against_exact_trace_and_pinned_source_before_H4_disposition",
            "schema_version": 1,
            "trace_sha256": "a" * 64,
        },
    )
    args = SimpleNamespace(
        h4_contract=CONTRACT, profile_validation=profile_path, reviewed_mapping=mapping_path, candidate_chunk_size=128
    )
    report = validate_kernel_audit(args)
    assert report["status"] == "passed_kernel_mapping_only_H4_set_validation_still_required"
    changed = json.loads(mapping_path.read_text())
    changed["accelerator_events"][0]["exact_name"] = "unobserved"
    _write_json(mapping_path, changed)
    with pytest.raises(ValueError, match="event set drift"):
        validate_kernel_audit(args)


def _make_h4_set_inputs(tmp_path: Path, *, unstable: bool = False):
    _, contract_sha256 = load_h4_contract(CONTRACT)
    candidate_args = []
    kernel_args = []
    for position, chunk_size in enumerate((128, 256, 512, 1024)):
        values = ([1.0, 20.0] * 5) if unstable and chunk_size == 128 else [10.0 + position * 0.05] * 10
        candidate_path = tmp_path / f"candidate-{chunk_size}.json"
        kernel_path = tmp_path / f"kernel-{chunk_size}.json"
        profile_validation_sha256 = f"{position + 1:064x}"
        _write_json(
            candidate_path,
            {
                "artifact": "qwen35_r18_h4_candidate_automated_validation",
                "candidate_chunk_size": chunk_size,
                "eligible_pending_manual_kernel_mapping": True,
                "h4_contract_sha256": contract_sha256,
                "measured_synchronized_update_seconds": values,
                "profile_validation_sha256": profile_validation_sha256,
                "qualification_manifest_sha256": "b" * 64,
                "slurm_account": "aifac_f02_434",
                "slurm_job_id": str(100 + position),
                "source_commit": "c" * 40,
                "status": "automated_candidate_passed_pending_manual_kernel_mapping",
                "timing_coefficient_of_variation_exceeds_threshold": unstable and chunk_size == 128,
                "timing_statistics": timing_statistics(values),
            },
        )
        _write_json(
            kernel_path,
            {
                "artifact": "qwen35_r18_h4_final_kernel_audit",
                "candidate_chunk_size": chunk_size,
                "h4_contract_sha256": contract_sha256,
                "profile_validation_sha256": profile_validation_sha256,
                "status": "passed_kernel_mapping_only_H4_set_validation_still_required",
            },
        )
        candidate_args.append((chunk_size, candidate_path))
        kernel_args.append((chunk_size, kernel_path))
    return SimpleNamespace(h4_contract=CONTRACT, candidate_report=candidate_args, kernel_audit=kernel_args)


def test_h4_set_validator_selects_only_after_all_four_independent_kernel_audits(tmp_path):
    report = validate_h4_set(_make_h4_set_inputs(tmp_path))
    assert report["status"] == "passed_H5_only_authorized"
    assert report["allowed_successor"] == "H5_only"
    assert report["scientific_training_authorized"] is False
    assert report["selection"]["selected_chunk_size"] == 128


def test_h4_set_validator_uses_frozen_complete_repeat_branch_for_unstable_primary_set(tmp_path):
    report = validate_h4_set(_make_h4_set_inputs(tmp_path, unstable=True))
    assert report["status"] == "timing_repeat_required_H4_not_passed"
    assert report["allowed_successor"] is None
    assert report["selection"] is None
    assert report["single_complete_four_candidate_timing_repeat_authorized"] is True
