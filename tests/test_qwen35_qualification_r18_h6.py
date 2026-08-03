import copy
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from scripts.train.qwen35.compare_qwen35_checkpoints import (
    EXPECTED_TRAINER_STATE_KEYS,
    compare_checkpoints,
    compare_nested,
    compare_trainer_state,
    deterministic_log_history,
    tensor_bit_equal,
    write_report_atomic,
)
from scripts.train.qwen35.validate_qwen35_h6_r18 import deterministic_metric_projection, validate_metrics

from open_instruct.qwen35_qualification_r18_h6 import (
    H6_CONTRACT_SHA256,
    H6_EXPECTED_TARGETS_BY_UPDATE,
    load_h6_contract,
    validate_h6_source_delta,
)

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parents[1]
CONTRACT = REPOSITORY / "scripts/train/qwen35/qwen35_hardware_qualification_r18_h6.json"
HUMAN_PROTOCOL = WORKSPACE / "methodology/qwen35_hardware_qualification_r18_h6_protocol_r1_20260720.md"
H5_FINAL = (
    WORKSPACE
    / "artifacts/qwen35_hardware_qualification_20260718/r18_h5_final_closure_20260720.json"
)
PREREGISTRATION = (
    WORKSPACE
    / "artifacts/qwen35_hardware_qualification_20260718/r18_h6_preregistration_closure_20260720.json"
)
H5_EVIDENCE = (
    WORKSPACE
    / "artifacts/qwen35_hardware_qualification_20260718/r18_h5_gpu_attempt03_49882653_evidence"
)
SCHEDULE = (
    WORKSPACE
    / "artifacts/qwen35_hardware_qualification_20260718/"
    "r18_h5_schedule_materialization_attempt02_49876428/"
    "qwen35_c00_seed3407_010steps_080packs.json"
)


def strict_json_copy(source: Path, target: Path, mutation=None) -> None:
    value = json.loads(source.read_text())
    if mutation is not None:
        mutation(value)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def trainer_state(*, observational_step5_summary: bool = False) -> dict:
    logs = [
        {
            "epoch": step / 10,
            "grad_norm": float(step),
            "learning_rate": float(11 - step) * 1e-6,
            "loss": float(step) / 100,
            "step": step,
        }
        for step in range(1, 11)
    ]
    if observational_step5_summary:
        logs.insert(
            5,
            {
                "epoch": 0.5,
                "step": 5,
                "total_flos": 0.0,
                "train_loss": 0.03,
                "train_runtime": 10.0,
                "train_samples_per_second": 4.0,
                "train_steps_per_second": 0.5,
            },
        )
    logs.append(
        {
            "epoch": 1.0,
            "step": 10,
            "total_flos": 0.0,
            "train_loss": 0.055,
            "train_runtime": 20.0 if not observational_step5_summary else 9.0,
            "train_samples_per_second": 4.0,
            "train_steps_per_second": 0.5,
        }
    )
    value = {
        "best_global_step": None,
        "best_metric": None,
        "best_model_checkpoint": None,
        "epoch": 1.0,
        "eval_steps": 500,
        "global_step": 10,
        "is_hyper_param_search": False,
        "is_local_process_zero": True,
        "is_world_process_zero": True,
        "log_history": logs,
        "logging_steps": 1.0,
        "max_steps": 10,
        "num_input_tokens_seen": 0,
        "num_train_epochs": 1,
        "save_steps": 10,
        "stateful_callbacks": {
            "TrainerControl": {
                "args": {
                    "should_epoch_stop": False,
                    "should_evaluate": False,
                    "should_log": False,
                    "should_save": True,
                    "should_training_stop": True,
                },
                "attributes": {},
            }
        },
        "total_flos": 0.0,
        "train_batch_size": 1,
        "trial_name": None,
        "trial_params": None,
    }
    assert set(value) == EXPECTED_TRAINER_STATE_KEYS
    return value


def make_checkpoint(path: Path, *, resumed: bool = False) -> None:
    path.mkdir()
    save_file(
        {
            "model.float": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "model.integer": torch.tensor([1, 2], dtype=torch.int64),
        },
        path / "model.safetensors",
    )
    torch.save(
        {
            "state": {
                0: {
                    "step": torch.tensor(10.0),
                    "exp_avg": torch.tensor([0.5]),
                    "exp_avg_sq": torch.tensor([0.25]),
                }
            },
            "param_groups": [{"lr": 1e-5, "capturable": False}],
        },
        path / "optimizer.pt",
    )
    torch.save({"base_lrs": [2e-5], "last_epoch": 10, "_step_count": 11}, path / "scheduler.pt")
    for rank in range(4):
        state = {
            "python": random.Random(rank).getstate(),
            "numpy": np.random.RandomState(rank).get_state(),
            "cpu": torch.Generator().manual_seed(rank).get_state(),
            "cuda": torch.Generator().manual_seed(100 + rank).get_state(),
        }
        torch.save(state, path / f"rng_state_{rank}.pth")
    (path / "trainer_state.json").write_text(
        json.dumps(trainer_state(observational_step5_summary=resumed), indent=2, sort_keys=True) + "\n"
    )


def test_h6_contract_and_authority_bindings_load_exactly():
    contract, digest = load_h6_contract(
        CONTRACT,
        human_protocol_path=HUMAN_PROTOCOL,
        h5_final_closure_path=H5_FINAL,
        preregistration_closure_path=PREREGISTRATION,
    )

    assert digest == H6_CONTRACT_SHA256
    assert contract["scientific_training_authorized"] is False
    assert contract["successor_on_complete_independent_pass"] == "H7_only"
    assert contract["exposure"]["per_update_assistant_targets"] == list(H6_EXPECTED_TARGETS_BY_UPDATE)


def test_h6_contract_rejects_even_semantically_plausible_mutation(tmp_path):
    mutated = tmp_path / "contract.json"
    strict_json_copy(CONTRACT, mutated, lambda value: value["comparison"].update({"atol": 1e-8}))

    with pytest.raises(ValueError, match="contract digest drift"):
        load_h6_contract(
            mutated,
            human_protocol_path=HUMAN_PROTOCOL,
            h5_final_closure_path=H5_FINAL,
            preregistration_closure_path=PREREGISTRATION,
        )


def test_h6_source_delta_is_clean_descendant_and_inside_exact_allowlist():
    head = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    report = validate_h6_source_delta(REPOSITORY, expected_head=head)

    assert report["status"] == "passed"
    assert set(report["observed_changed_paths"]) == set(report["allowed_paths"])


def test_checkpoint_comparator_covers_model_optimizer_scheduler_rng_and_trainer(tmp_path):
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    make_checkpoint(continuous)
    make_checkpoint(resumed, resumed=True)

    report = compare_checkpoints(continuous, resumed, atol=0.0, rtol=0.0)

    assert report["status"] == "passed"
    assert report["model"]["bit_exact"] is True
    assert report["optimizer"]["bit_exact_tensors"] is True
    assert report["scheduler"]["bit_exact_tensors"] is True
    assert set(report["rng"]) == {f"rng_state_{rank}.pth" for rank in range(4)}
    assert report["trainer_state"]["deterministic_log_history_length"] == 10
    assert report["trainer_state"]["raw_log_history_lengths"] == [11, 12]


@pytest.mark.parametrize("artifact", ["model", "optimizer", "scheduler", "rng", "trainer"])
def test_checkpoint_comparator_rejects_each_semantic_state_class(tmp_path, artifact):
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    make_checkpoint(continuous)
    make_checkpoint(resumed, resumed=True)
    if artifact == "model":
        save_file(
            {
                "model.float": torch.tensor([1.0, 2.1]),
                "model.integer": torch.tensor([1, 2], dtype=torch.int64),
            },
            resumed / "model.safetensors",
        )
    elif artifact == "optimizer":
        value = torch.load(resumed / "optimizer.pt", weights_only=True)
        value["state"][0]["exp_avg"][0] += 1
        torch.save(value, resumed / "optimizer.pt")
    elif artifact == "scheduler":
        value = torch.load(resumed / "scheduler.pt", weights_only=True)
        value["last_epoch"] = 9
        torch.save(value, resumed / "scheduler.pt")
    elif artifact == "rng":
        value = torch.load(resumed / "rng_state_2.pth", weights_only=False)  # test-created trusted file
        value["cpu"][0] ^= 1
        torch.save(value, resumed / "rng_state_2.pth")
    else:
        value = json.loads((resumed / "trainer_state.json").read_text())
        value["log_history"][6]["loss"] += 0.01
        (resumed / "trainer_state.json").write_text(json.dumps(value))

    with pytest.raises((AssertionError, ValueError)):
        compare_checkpoints(continuous, resumed, atol=0.0, rtol=0.0)


def test_checkpoint_comparator_rejects_scalar_type_rng_file_set_and_unknown_trainer_shape(tmp_path):
    counters = {"tensors": 0, "nonidentical_tensors": 0, "numpy_arrays": 0, "scalars": 0}
    with pytest.raises(ValueError, match="scalar type mismatch"):
        compare_nested(False, 0, path="state.flag", atol=0, rtol=0, counters=counters)
    assert tensor_bit_equal(torch.tensor(0.0), torch.tensor(-0.0)) is False
    assert tensor_bit_equal(torch.tensor(0.0), torch.tensor(0.0)) is True

    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    make_checkpoint(continuous)
    make_checkpoint(resumed, resumed=True)
    (resumed / "rng_state_3.pth").unlink()
    with pytest.raises(ValueError, match="RNG-state file sets differ"):
        compare_checkpoints(continuous, resumed, atol=0.0, rtol=0.0)

    unknown = trainer_state()
    unknown["log_history"][0]["new_observational_or_semantic_field"] = 1
    with pytest.raises(ValueError, match="unexpected trainer log_history"):
        deterministic_log_history(unknown)


def test_strict_trainer_json_rejects_duplicate_keys_and_atomic_writer_refuses_overwrite(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text('{"global_step":10,"global_step":10}\n')
    right.write_text(json.dumps(trainer_state()))
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        compare_trainer_state(left, right, strict_h6=True)

    output = tmp_path / "report.json"
    write_report_atomic(output, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_report_atomic(output, {"status": "silently_overwritten"})
    assert json.loads(output.read_text()) == {"status": "passed"}


def test_deterministic_metric_projection_excludes_only_observational_measurements():
    record = json.loads((H5_EVIDENCE / "qwen35_exact_metrics.jsonl").read_text().splitlines()[0])
    baseline = deterministic_metric_projection(record)
    observational = copy.deepcopy(record)
    observational["elapsed_seconds"] = 1e9
    observational["rates"] = {"fabricated": 0}
    observational["memory"] = {"peak_allocated_bytes": 1}
    observational["timing"] = {"host": "different"}
    observational["analytic_flops"]["analytic_model_mfu"] = 0.99
    observational["analytic_flops"]["caveat"] = "different"
    assert deterministic_metric_projection(observational) == baseline

    semantic = copy.deepcopy(record)
    semantic["loss"]["normalized_loss"] += 1e-12
    assert deterministic_metric_projection(semantic) != baseline


def test_h5_metrics_prefix_passes_h6_validator_and_semantic_drift_fails(tmp_path):
    metrics = H5_EVIDENCE / "qwen35_exact_metrics.jsonl"
    summary = H5_EVIDENCE / "qwen35_exact_metrics_summary.json"
    schedule_entries = json.loads(SCHEDULE.read_text())["entries"]
    report = validate_metrics(
        metrics, summary, first_step=1, last_step=5, schedule_entries=schedule_entries
    )
    assert len(report["projections"]) == 5

    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows[2]["schedule_indices"][0] += 1
    drifted = tmp_path / "metrics.jsonl"
    drifted.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="schedule-index exposure drift"):
        validate_metrics(
            drifted,
            summary,
            first_step=1,
            last_step=5,
            schedule_entries=schedule_entries,
        )


def test_h6_wrapper_is_bounded_personal_c00_only_and_has_no_launch_chaining():
    wrapper = (REPOSITORY / "scripts/train/qwen35/leonardo_h6_r18.sbatch").read_text()

    assert "#SBATCH --account=aifac_f02_434" in wrapper
    assert "#SBATCH --gres=gpu:4" in wrapper
    assert "#SBATCH --time=00:45:00" in wrapper
    assert wrapper.count('"$QWEN35_VENV/bin/torchrun"') == 2
    assert "--expected_arm_id C00" in wrapper
    assert "--hardware_profile false" in wrapper
    assert "--cuda_event_step_timing false" in wrapper
    assert "--require_no_dense_logits true" in wrapper
    assert "--require_forward_loss_audit true" in wrapper
    assert "--use_liger_fused_linear_cross_entropy false" in wrapper
    assert "--selected_loss_chunk_size 512" in wrapper
    assert "--atol 0" in wrapper and "--rtol 0" in wrapper
    assert "R18_H6_PRODUCER_PASSED_PENDING_SLURM_AND_INDEPENDENT_CLOSURE" in wrapper
    assert "sbatch " not in wrapper
    assert "C01" not in wrapper and "BFCL" not in wrapper and "tau2" not in wrapper
