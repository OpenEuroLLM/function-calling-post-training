from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.train.qwen35.capture_qwen35_hardware_inventory import expected_runtime_versions

from open_instruct.qwen35_qualification_r17 import load_qualification_manifest as load_r17
from open_instruct.qwen35_qualification_r18 import load_qualification_manifest

ROOT = Path(__file__).parents[1]
R18 = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r18.json"
R17 = ROOT / "scripts/train/qwen35/qwen35_hardware_qualification_r17.json"


def test_r18_h0_runtime_contract_excludes_liger_and_fails_closed_without_explicit_ban() -> None:
    qualification, _ = load_qualification_manifest(R18)
    versions = expected_runtime_versions(qualification["runtime_pins"])
    assert "liger-kernel" not in versions
    assert qualification["runtime_pins"]["liger_execution_allowed"] is False
    mutated = dict(qualification["runtime_pins"])
    mutated.pop("liger_execution_allowed")
    with pytest.raises(RuntimeError, match="explicitly forbid"):
        expected_runtime_versions(mutated)


def test_historical_r17_h0_runtime_contract_still_pins_liger() -> None:
    qualification, _ = load_r17(R17)
    versions = expected_runtime_versions(qualification["runtime_pins"])
    assert versions["liger-kernel"] == qualification["runtime_pins"]["liger_version"]


def test_importing_h0_module_in_fresh_process_does_not_import_liger() -> None:
    code = """
import json
import sys
import scripts.train.qwen35.capture_qwen35_hardware_inventory
print(json.dumps(sorted(name for name in sys.modules if name == 'liger_kernel' or name.startswith('liger_kernel.'))))
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == []
