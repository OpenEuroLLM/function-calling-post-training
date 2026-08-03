import json
import random

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from scripts.train.qwen35.compare_qwen35_checkpoints import (
    compare_model,
    compare_torch_artifact,
    compare_trainer_state,
    safe_torch_load,
)


def _checkpoint(path, *, model_delta=0.0, optimizer_delta=0.0, global_step=10):
    path.mkdir()
    save_file(
        {"model.float": torch.tensor([1.0 + model_delta, 2.0]), "model.integer": torch.tensor([1, 2])},
        path / "model.safetensors",
    )
    torch.save(
        {
            "state": {
                0: {
                    "step": torch.tensor(10.0),
                    "exp_avg": torch.tensor([0.5 + optimizer_delta]),
                    "exp_avg_sq": torch.tensor([0.25]),
                }
            },
            "param_groups": [{"lr": 2e-5}],
        },
        path / "optimizer.pt",
    )
    (path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "max_steps": 10,
                "num_train_epochs": 1,
                "train_batch_size": 4,
                "trial_name": None,
                "trial_params": None,
                "log_history": [],
            }
        )
    )


def test_checkpoint_comparison_proves_semantic_bit_identity(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _checkpoint(left)
    _checkpoint(right)

    model = compare_model(left, right, atol=0, rtol=0)
    optimizer = compare_torch_artifact(left / "optimizer.pt", right / "optimizer.pt", atol=0, rtol=0)
    trainer = compare_trainer_state(left / "trainer_state.json", right / "trainer_state.json")

    assert model["bit_exact"] is True
    assert model["nonidentical_tensors"] == 0
    assert optimizer["bit_exact_tensors"] is True
    assert trainer["global_step"] == 10


def test_checkpoint_comparison_rejects_model_optimizer_and_state_drift(tmp_path):
    left = tmp_path / "left"
    model_drift = tmp_path / "model-drift"
    optimizer_drift = tmp_path / "optimizer-drift"
    step_drift = tmp_path / "step-drift"
    _checkpoint(left)
    _checkpoint(model_drift, model_delta=0.1)
    _checkpoint(optimizer_drift, optimizer_delta=0.1)
    _checkpoint(step_drift, global_step=9)

    with pytest.raises(AssertionError):
        compare_model(left, model_drift, atol=0, rtol=0)
    with pytest.raises(AssertionError):
        compare_torch_artifact(left / "optimizer.pt", optimizer_drift / "optimizer.pt", atol=0, rtol=0)
    with pytest.raises(AssertionError, match="semantic fields"):
        compare_trainer_state(left / "trainer_state.json", step_drift / "trainer_state.json")


def test_checkpoint_comparison_accepts_only_explicit_nonzero_tolerance(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _checkpoint(left)
    _checkpoint(right, model_delta=1e-5)

    report = compare_model(left, right, atol=2e-5, rtol=0)

    assert report["bit_exact"] is False
    assert report["maximum_absolute_error"] == pytest.approx(1e-5, rel=1e-2)


def test_safe_loader_accepts_hf_rng_state_without_unrestricted_pickle(tmp_path):
    rng_state = {
        "python": random.Random(7).getstate(),
        "numpy": np.random.RandomState(7).get_state(),
        "cpu": torch.Generator().manual_seed(7).get_state(),
    }
    path = tmp_path / "rng_state.pth"
    torch.save(rng_state, path)

    loaded = safe_torch_load(path)

    assert loaded["python"] == rng_state["python"]
    assert np.array_equal(loaded["numpy"][1], rng_state["numpy"][1])
    assert torch.equal(loaded["cpu"], rng_state["cpu"])
