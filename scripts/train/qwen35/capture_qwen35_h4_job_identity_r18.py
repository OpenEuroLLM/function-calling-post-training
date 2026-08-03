#!/usr/bin/env python3
"""Capture immutable source, data, model, runtime, GPU, and Slurm identity for one H4 candidate job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_qualification_r18_h4 import (
    LEONARDO_A100_COMPUTE_CAPABILITY,
    LEONARDO_A100_MEMORY_MIB,
    LEONARDO_A100_NAME,
    load_h4_contract,
    load_strict_json,
    sha256_file,
)
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--numpy-data", type=Path, required=True)
    parser.add_argument("--four-update-schedule", type=Path, required=True)
    parser.add_argument("--thirteen-update-schedule", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--h4-contract", type=Path, required=True)
    parser.add_argument("--candidate-chunk-size", type=int, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _nvidia_inventory() -> list[dict[str, str]]:
    fields = ["index", "name", "uuid", "memory.total", "driver_version", "compute_cap"]
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise ValueError(f"unexpected nvidia-smi inventory row: {line!r}")
        rows.append(dict(zip(fields, values, strict=True)))
    if len(rows) != 1:
        raise ValueError(f"H4 one-GPU job observed {len(rows)} visible GPUs")
    expected_identity = {
        "compute_cap": LEONARDO_A100_COMPUTE_CAPABILITY,
        "memory.total": LEONARDO_A100_MEMORY_MIB,
        "name": LEONARDO_A100_NAME,
    }
    if any(rows[0][field] != expected for field, expected in expected_identity.items()):
        raise ValueError(f"H4 unexpected visible GPU identity: {rows[0]}")
    return rows


def main() -> int:
    args = parse_args()
    if args.report_output.exists():
        raise FileExistsError(args.report_output)
    if os.environ.get("SLURM_JOB_ACCOUNT") != "aifac_f02_434":
        raise RuntimeError("H4 identity capture refuses a non-personal Slurm account")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("H4 identity capture requires a Slurm allocation")
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    h4, h4_sha256 = load_h4_contract(args.h4_contract)
    if qualification_sha256 != h4["parent"]["r18_machine_manifest_sha256"]:
        raise ValueError("H4 identity capture R18/H4 identity drift")
    if args.candidate_chunk_size not in h4["candidate_chunk_sizes_in_execution_order"]:
        raise ValueError("H4 identity capture received an unknown chunk candidate")

    repo = args.repo_root.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.expected_code_commit:
        raise ValueError(f"H4 staged source commit drift: {commit} != {args.expected_code_commit}")
    if _git(repo, "status", "--porcelain"):
        raise ValueError("H4 staged source worktree is dirty")
    pyc = sorted(str(path.relative_to(repo)) for path in repo.rglob("*.pyc"))
    if pyc:
        raise ValueError(f"H4 staged source tree contains Python bytecode: {pyc[:10]}")
    subprocess.run(
        ["sha256sum", "--check", "--strict", str(args.code_manifest.resolve())], cwd=repo, check=True
    )
    subprocess.run(
        ["sha256sum", "--check", "--strict", str(args.model_manifest.resolve())],
        cwd=args.model_snapshot.resolve(),
        check=True,
    )
    numpy_manifest = args.numpy_data / "manifest.json"
    runtime_report = load_strict_json(args.runtime_report)
    if runtime_report.get("status") != "passed":
        raise ValueError("H4 runtime import report did not pass")
    files = {
        "code_manifest": args.code_manifest,
        "four_update_schedule": args.four_update_schedule,
        "h4_contract": args.h4_contract,
        "model_manifest": args.model_manifest,
        "numpy_manifest": numpy_manifest,
        "qualification_manifest": args.qualification_manifest,
        "runtime_report": args.runtime_report,
        "thirteen_update_schedule": args.thirteen_update_schedule,
    }
    for label, path in files.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"H4 identity input {label} is absent or empty: {path}")
    if sha256_file(numpy_manifest) != h4["data"]["numpy_manifest_sha256"]:
        raise ValueError("H4 identity capture C00 NumPy manifest drift")
    if sha256_file(args.four_update_schedule) != h4["four_update_schedule"]["file_sha256"]:
        raise ValueError("H4 identity capture four-update schedule drift")
    if sha256_file(args.thirteen_update_schedule) != h4["thirteen_update_schedule"]["file_sha256"]:
        raise ValueError("H4 identity capture thirteen-update schedule drift")
    report = {
        "artifact": "qwen35_r18_h4_candidate_job_identity",
        "candidate_chunk_size": args.candidate_chunk_size,
        "environment": {
            "pythonpath": os.environ.get("PYTHONPATH"),
            "runtime_report_sha256": sha256_file(args.runtime_report),
        },
        "file_identities": {
            label: {"bytes": path.stat().st_size, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in sorted(files.items())
        },
        "git": {
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": commit,
            "status_porcelain": "",
            "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        },
        "gpu_inventory": _nvidia_inventory(),
        "h4_contract_sha256": h4_sha256,
        "model_snapshot": str(args.model_snapshot.resolve()),
        "qualification_manifest_sha256": qualification_sha256,
        "schema_version": 1,
        "scientific_training_authorized": False,
        "slurm": {
            "account": os.environ["SLURM_JOB_ACCOUNT"],
            "job_id": os.environ["SLURM_JOB_ID"],
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "source_python_bytecode_files": [],
        "status": "passed_identity_capture_only",
    }
    write_json_atomic(args.report_output, report)
    print(json.dumps({"output": str(args.report_output), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
