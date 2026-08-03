#!/usr/bin/env python3
"""Independently validate the complete eight-run R18 H3 evidence set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification import sha256_file
from open_instruct.qwen35_qualification_r18_h3 import load_h3_harness_amendment, load_h3_manifest, validate_h3_report

VALIDATOR_AMENDMENT_ID = "qwen35-r18-h3-independent-validator-amendment-v1"
VALIDATOR_AMENDMENT_SHA256 = "bc648147a2af821544fc30554186c05b0758cb9780bf58887eae88f766b66644"
VALIDATOR_AMENDMENT_HUMAN_SHA256 = "1934d64a3714648a4144f36a1cd2cd445a46ce68e08cab733a2c10fd5daca81d"
VALIDATOR_AMENDMENT_PREREGISTRATION_SHA256 = "d9ce018a7c110ad550ce6523a21835a4bcd48c42614492e6179ffe780c63f6ef"
VALIDATOR_CONSISTENCY_AMENDMENT_ID = "qwen35-r18-h3-independent-validator-consistency-amendment-v3"
VALIDATOR_CONSISTENCY_AMENDMENT_SHA256 = "1090502a52feeed6cbf878018a2c251f02ce0a1431f8974b4e59ea13be950dcf"
VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_SHA256 = (
    "90d26f74ef51c1ccee8cdbc5e75ae3708a172bc65918912295c07edfd70cc671"
)
VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION_SHA256 = (
    "77e09a4b862e58bd496bfe839eee0653724914fed5ac4449ed12a901a021c27e"
)
VALIDATOR_V2_FAILURE_CLOSURE_SHA256 = "9946450917786e13cb297b319769d06aa302c0a3b2e62eda5b43df59b69011aa"
VALIDATOR_FAILURE_FORENSIC_CLOSURE_SHA256 = (
    "497be6ced00350270cf77dfe89a15313c1f451b5e416985d12fba4533712e1a4"
)
H2_PREDECESSOR_SHA256 = "cd01fe72f7f73c3d2496390a8ccbee2718f78b7843208ccc626d1c7a51fed176"
R18_MANIFEST_SHA256 = "679ad710f0be07f811071b1a56863b8cb851732a0ac8a808f4e5747e9c325ee0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3-manifest", type=Path, required=True)
    parser.add_argument("--r18-manifest", type=Path, required=True)
    parser.add_argument("--human-protocol", type=Path, required=True)
    parser.add_argument("--preregistration-closure", type=Path, required=True)
    parser.add_argument("--harness-amendment", type=Path, required=True)
    parser.add_argument("--harness-amendment-human-protocol", type=Path, required=True)
    parser.add_argument("--harness-amendment-preregistration-closure", type=Path, required=True)
    parser.add_argument("--attempt01-failure-closure", type=Path, required=True)
    parser.add_argument("--validator-amendment", type=Path, required=True)
    parser.add_argument("--validator-amendment-human-protocol", type=Path, required=True)
    parser.add_argument("--validator-amendment-preregistration-closure", type=Path, required=True)
    parser.add_argument("--validator-consistency-amendment", type=Path, required=True)
    parser.add_argument("--validator-consistency-amendment-human-protocol", type=Path, required=True)
    parser.add_argument("--validator-consistency-amendment-preregistration-closure", type=Path, required=True)
    parser.add_argument("--validator-v2-failure-closure", type=Path, required=True)
    parser.add_argument("--validator-failure-forensic-closure", type=Path, required=True)
    parser.add_argument("--h2-independent-validation", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser.parse_args()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def git_output(root: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=not binary).stdout


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_plain_int(value: Any, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"H2 predecessor {label} drift")


def validate_h2_predecessor(path: Path) -> dict[str, Any]:
    """Validate the exact authoritative H2 artifact and its real field names."""

    if sha256_file(path) != H2_PREDECESSOR_SHA256:
        raise ValueError("H3 predecessor H2 independent-validation hash drift")
    value = json.loads(path.read_text())
    validate_h2_predecessor_value(value)
    return value


def validate_h2_predecessor_value(value: dict[str, Any]) -> None:
    """Validate semantic H2 fields separately so adversarial tests cover each guard."""

    if "allowed_successor" in value:
        raise ValueError("H2 predecessor contains the stale allowed_successor key")
    if value.get("artifact") != "qwen35_h2_r18_independent_validation_amended_v1":
        raise ValueError("H2 predecessor artifact identity drift")
    _require_plain_int(value.get("schema_version"), 1, label="schema version")
    if value.get("status") != "passed":
        raise ValueError("H2 predecessor status does not authorize H3")
    if value.get("successor_gate_authorized") != "H3_only":
        raise ValueError("H2 predecessor successor authorization does not authorize H3")
    if value.get("scientific_training_authorized") is not False:
        raise ValueError("H2 predecessor scientific-training authority drift")
    if value.get("qualification_manifest_sha256") != R18_MANIFEST_SHA256:
        raise ValueError("H2 predecessor qualification-manifest binding drift")
    validation = value.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("H2 predecessor nested validation is missing")
    if validation.get("status") != "passed":
        raise ValueError("H2 predecessor nested validation did not pass")
    if validation.get("successor_gate_authorized") is not True:
        raise ValueError("H2 predecessor nested successor authorization drift")
    if validation.get("scientific_training_authorized") is not False:
        raise ValueError("H2 predecessor nested scientific-training authority drift")
    _require_plain_int(validation.get("candidate_count"), 4, label="candidate count")
    _require_plain_int(validation.get("trajectory_steps"), 3072, label="trajectory-step count")


def validate_validator_amendment(
    machine_path: Path, *, human_path: Path, preregistration_path: Path
) -> tuple[dict[str, Any], str]:
    """Resolve the preregistered validator-only correction and all bound bytes."""

    digest = sha256_file(machine_path)
    if digest != VALIDATOR_AMENDMENT_SHA256:
        raise ValueError("H3 validator-amendment machine-manifest hash drift")
    amendment = json.loads(machine_path.read_text())
    if amendment.get("amendment_id") != VALIDATOR_AMENDMENT_ID:
        raise ValueError("H3 validator-amendment identity drift")
    if amendment.get("status") != "preregistered_before_corrected_validator_implementation_or_execution":
        raise ValueError("H3 validator-amendment status drift")
    if amendment.get("human_amendment") != {
        "path": "methodology/qwen35_hardware_qualification_r18_h3_validator_amendment_v1_20260719.md",
        "sha256": VALIDATOR_AMENDMENT_HUMAN_SHA256,
    }:
        raise ValueError("H3 validator-amendment human binding drift")
    if sha256_file(human_path) != VALIDATOR_AMENDMENT_HUMAN_SHA256:
        raise ValueError("H3 validator-amendment human file hash drift")
    expected_h2 = amendment.get("authoritative_h2_predecessor", {})
    if (
        expected_h2.get("sha256") != H2_PREDECESSOR_SHA256
        or expected_h2.get("successor_gate_authorized") != "H3_only"
        or expected_h2.get("forbidden_top_level_keys") != ["allowed_successor"]
    ):
        raise ValueError("H3 validator-amendment predecessor contract drift")
    if amendment.get("scientific_training_authorized") is not False:
        raise ValueError("H3 validator-amendment scientific-training authority drift")
    if amendment.get("allowed_successor_on_complete_pass") != "H4_only":
        raise ValueError("H3 validator-amendment successor drift")

    if sha256_file(preregistration_path) != VALIDATOR_AMENDMENT_PREREGISTRATION_SHA256:
        raise ValueError("H3 validator-amendment preregistration-closure hash drift")
    closure = json.loads(preregistration_path.read_text())
    if closure.get("artifact") != "qwen35_r18_h3_validator_amendment_v1_preregistration_closure":
        raise ValueError("H3 validator-amendment preregistration identity drift")
    if closure.get("status") != "closed_before_corrected_validator_implementation_or_execution":
        raise ValueError("H3 validator-amendment preregistration status drift")
    if closure.get("human_amendment", {}).get("sha256") != VALIDATOR_AMENDMENT_HUMAN_SHA256:
        raise ValueError("H3 validator-amendment preregistration human binding drift")
    if closure.get("machine_manifest", {}).get("sha256") != VALIDATOR_AMENDMENT_SHA256:
        raise ValueError("H3 validator-amendment preregistration machine binding drift")
    if closure.get("machine_manifest", {}).get("git_blob_commit") != (
        "0cd67bd50ed564998e3e92d8f9e522516ec2493f"
    ):
        raise ValueError("H3 validator-amendment preregistration commit drift")
    if closure.get("authoritative_h2_predecessor_sha256") != H2_PREDECESSOR_SHA256:
        raise ValueError("H3 validator-amendment preregistration predecessor binding drift")
    return amendment, digest


def validate_validator_consistency_amendment(
    machine_path: Path,
    *,
    human_path: Path,
    preregistration_path: Path,
    validator_failure_path: Path,
    forensic_closure_path: Path,
) -> tuple[dict[str, Any], str]:
    """Bind the post-outcome, mathematically derived cross-backend summary correction."""

    digest = sha256_file(machine_path)
    if digest != VALIDATOR_CONSISTENCY_AMENDMENT_SHA256:
        raise ValueError("H3 validator-consistency amendment machine-manifest hash drift")
    amendment = json.loads(machine_path.read_text())
    if amendment.get("amendment_id") != VALIDATOR_CONSISTENCY_AMENDMENT_ID:
        raise ValueError("H3 validator-consistency amendment identity drift")
    if amendment.get("status") != "preregistered_after_diagnosis_before_amended_implementation_or_execution":
        raise ValueError("H3 validator-consistency amendment status drift")
    if amendment.get("human_amendment") != {
        "path": "methodology/qwen35_hardware_qualification_r18_h3_validator_consistency_amendment_v3_20260719.md",
        "sha256": VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_SHA256,
    }:
        raise ValueError("H3 validator-consistency human binding drift")
    if sha256_file(human_path) != VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_SHA256:
        raise ValueError("H3 validator-consistency human file hash drift")
    if amendment.get("failed_validator") != {
        "failure_closure_sha256": VALIDATOR_V2_FAILURE_CLOSURE_SHA256,
        "job_id": "49858377",
        "status": "failed_strict_independent_validation",
    }:
        raise ValueError("H3 validator-consistency failure binding drift")
    if sha256_file(validator_failure_path) != VALIDATOR_V2_FAILURE_CLOSURE_SHA256:
        raise ValueError("H3 validator V2 failure-closure hash drift")
    failure = json.loads(validator_failure_path.read_text())
    if (
        failure.get("status") != "failed_strict_independent_validation"
        or failure.get("diagnosis", {}).get("validator_entry_point_invocations") != 1
    ):
        raise ValueError("H3 validator V2 failure-closure semantic drift")
    if amendment.get("forensic", {}).get("closure_sha256") != VALIDATOR_FAILURE_FORENSIC_CLOSURE_SHA256:
        raise ValueError("H3 validator-consistency forensic binding drift")
    if sha256_file(forensic_closure_path) != VALIDATOR_FAILURE_FORENSIC_CLOSURE_SHA256:
        raise ValueError("H3 validator failure-forensic closure hash drift")
    forensic = json.loads(forensic_closure_path.read_text())
    if (
        forensic.get("status") != "diagnostic_complete"
        or forensic.get("findings", {}).get("scenario_candidate_runs") != 8
        or forensic.get("findings", {}).get("tensor_keys_examined") != 3888
        or forensic.get("scientific_training_authorized") is not False
    ):
        raise ValueError("H3 validator failure-forensic closure semantic drift")
    if amendment.get("bound") != {
        "binary64_unit_roundoff": "2^-53",
        "element_count": 103672,
        "formula": (
            "B_pair=2*b/(1-b); b=max((1+u)*sqrt(1+epsilon_sum)-1, "
            "1-(1-u)*sqrt(1-epsilon_sum)); epsilon_sum=(1+u)*(1+gamma_(n-1))-1; "
            "gamma_(n-1)=((n-1)*u)/(1-(n-1)*u)"
        ),
        "high_precision_decimal": (
            "1.1510126185730684230549368235693935884182021079188570533453842551364243845533590E-11"
        ),
        "minimum_decimal_precision": 80,
        "upward_rounded_binary64": 1.1510126185730685e-11,
    }:
        raise ValueError("H3 validator-consistency rounding bound drift")
    if (
        amendment.get("required_summary_comparisons") != 48
        or amendment.get("retry_limit") != 1
        or amendment.get("scientific_training_authorized") is not False
        or amendment.get("allowed_successor_on_complete_pass") != "H4_only"
    ):
        raise ValueError("H3 validator-consistency scope or authority drift")

    if sha256_file(preregistration_path) != VALIDATOR_CONSISTENCY_AMENDMENT_PREREGISTRATION_SHA256:
        raise ValueError("H3 validator-consistency preregistration-closure hash drift")
    closure = json.loads(preregistration_path.read_text())
    if (
        closure.get("artifact")
        != "qwen35_r18_h3_validator_consistency_amendment_v3_preregistration_closure"
        or closure.get("status") != "closed_before_amended_implementation_or_execution"
        or closure.get("machine_manifest", {}).get("git_blob_commit")
        != "ca954b2ecefc92139536b2477f3cfa9ad7e1ef8f"
        or closure.get("machine_manifest", {}).get("sha256") != VALIDATOR_CONSISTENCY_AMENDMENT_SHA256
        or closure.get("human_amendment", {}).get("sha256") != VALIDATOR_CONSISTENCY_AMENDMENT_HUMAN_SHA256
        or closure.get("failed_validator_closure_sha256") != VALIDATOR_V2_FAILURE_CLOSURE_SHA256
        or closure.get("forensic_closure_sha256") != VALIDATOR_FAILURE_FORENSIC_CLOSURE_SHA256
        or closure.get("scientific_training_authorized") is not False
    ):
        raise ValueError("H3 validator-consistency preregistration content drift")
    return amendment, digest


def validate_preregistration_closure(
    closure_path: Path, *, h3_manifest_path: Path, human_protocol_path: Path, h3_digest: str
) -> dict[str, Any]:
    closure = json.loads(closure_path.read_text())
    if closure != {
        "artifact": "qwen35_r18_h3_r1_preregistration_closure",
        "closed_at_utc": "2026-07-19T19:05:21Z",
        "human_protocol": {
            "path": "methodology/qwen35_hardware_qualification_r18_h3_protocol_r1_20260719.md",
            "sha256": "a50aaeecdb17a48f431902127d7d80b6760df6b1b93a7081c89607902522aaab",
        },
        "machine_manifest": {
            "git_blob_commit": "619edfddfadae58c53b5c71858a406c51e107921",
            "path": "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3.json",
            "sha256": "95aec699d2bab81c5eb3094d2048f997f137faa624dbb7128f92b32134b8abf4",
        },
        "parent_h2_independent_validation_sha256": "cd01fe72f7f73c3d2496390a8ccbee2718f78b7843208ccc626d1c7a51fed176",
        "preregistration_baseline_commit": "00fd47ec560ce7800c28b61b54451b803e17c9d9",
        "protocol_id": "qwen35-hardware-qualification-r18-h3-r1",
        "schema_version": 1,
        "statement": (
            "The human H3 protocol and machine H3 manifest were frozen before H3 implementation output and "
            "before any H3 CUDA execution."
        ),
        "status": "closed_before_implementation_and_execution",
    }:
        raise ValueError("H3 preregistration-closure content drift")
    if closure["machine_manifest"]["sha256"] != h3_digest or sha256_file(h3_manifest_path) != h3_digest:
        raise ValueError("H3 preregistration closure does not bind the machine manifest")
    if closure["human_protocol"]["sha256"] != sha256_file(human_protocol_path):
        raise ValueError("H3 preregistration closure does not bind the human protocol")
    return closure


def validate_producer_source(report: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    source = report["source_attestation"]
    commit = source["git_commit"]
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("H3 producer Git commit is malformed")
    if str(git_output(repo_root, "cat-file", "-t", commit)).strip() != "commit":
        raise ValueError("H3 producer Git commit is unavailable")
    validated = {}
    for relative, expected_digest in source["source_files_sha256"].items():
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("H3 producer source path escapes the repository")
        content = git_output(repo_root, "show", f"{commit}:{relative}", binary=True)
        digest = sha256_bytes(content)
        if digest != expected_digest:
            raise ValueError(f"H3 producer Git-object/source hash drift for {relative}")
        validated[relative] = digest
    required = {
        "open_instruct/qwen35_chunked_loss.py",
        "open_instruct/qwen35_qualification_r18_h3.py",
        "scripts/train/qwen35/g2_job_guard.sh",
        "scripts/train/qwen35/leonardo_h3_r18.sbatch",
        "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3.json",
        "scripts/train/qwen35/qwen35_hardware_qualification_r18_h3_harness_amendment_r2.json",
        "scripts/train/qwen35/validate_qwen35_ddp_ga_r18_h3.py",
    }
    if set(validated) != required:
        raise ValueError("H3 producer source-attestation file set drift")
    return {"git_commit": commit, "git_object_files": validated}


def main() -> None:
    args = parse_args()
    if str(git_output(args.repo_root, "status", "--porcelain")).strip():
        raise RuntimeError("H3 independent validation requires a clean source worktree")
    current_commit = str(git_output(args.repo_root, "rev-parse", "HEAD")).strip()
    h3, h3_digest, _ = load_h3_manifest(
        args.h3_manifest, r18_manifest_path=args.r18_manifest, human_protocol_path=args.human_protocol
    )
    validate_preregistration_closure(
        args.preregistration_closure,
        h3_manifest_path=args.h3_manifest,
        human_protocol_path=args.human_protocol,
        h3_digest=h3_digest,
    )
    harness_amendment, harness_amendment_digest = load_h3_harness_amendment(
        args.harness_amendment,
        human_amendment_path=args.harness_amendment_human_protocol,
        attempt01_failure_closure_path=args.attempt01_failure_closure,
        preregistration_closure_path=args.harness_amendment_preregistration_closure,
        h3_manifest_path=args.h3_manifest,
    )
    validator_amendment, validator_amendment_digest = validate_validator_amendment(
        args.validator_amendment,
        human_path=args.validator_amendment_human_protocol,
        preregistration_path=args.validator_amendment_preregistration_closure,
    )
    validator_consistency_amendment, validator_consistency_amendment_digest = (
        validate_validator_consistency_amendment(
            args.validator_consistency_amendment,
            human_path=args.validator_consistency_amendment_human_protocol,
            preregistration_path=args.validator_consistency_amendment_preregistration_closure,
            validator_failure_path=args.validator_v2_failure_closure,
            forensic_closure_path=args.validator_failure_forensic_closure,
        )
    )
    if h3["parent"]["h2_independent_validation_sha256"] != H2_PREDECESSOR_SHA256:
        raise ValueError("H3 predecessor H2 independent-validation hash drift")
    h2_validation = validate_h2_predecessor(args.h2_independent_validation)
    subprocess.run(
        ["sha256sum", "--check", "--strict", str(args.code_manifest)],
        cwd=args.repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    validations = []
    report_bindings = []
    producer_source = None
    producer_commit = None
    for scenario in h3["scenarios"]:
        scenario_id = scenario["scenario_id"]
        for chunk_size in h3["candidate_chunk_sizes_in_execution_order"]:
            run_dir = args.producer_root / f"{scenario_id}_c{chunk_size:04d}"
            report_path = run_dir / "h3_report.json"
            evidence_path = run_dir / "h3_evidence.safetensors"
            if not report_path.is_file() or not evidence_path.is_file():
                raise ValueError(f"missing H3 artifacts for {scenario_id}/C={chunk_size}")
            report = json.loads(report_path.read_text())
            validation = validate_h3_report(
                report,
                evidence_path=evidence_path,
                h3_manifest=h3,
                h3_manifest_sha256=h3_digest,
                r18_manifest_sha256=sha256_file(args.r18_manifest),
                require_pass=True,
            )
            if validation["scenario_id"] != scenario_id or validation["chunk_size"] != chunk_size:
                raise ValueError("H3 run-directory/report identity drift")
            source_validation = validate_producer_source(report, repo_root=args.repo_root)
            if producer_commit is None:
                producer_commit = source_validation["git_commit"]
                producer_source = source_validation
            elif source_validation != producer_source:
                raise ValueError("H3 producer source identity differs across the eight runs")
            validations.append(validation)
            report_bindings.append(
                {
                    "chunk_size": chunk_size,
                    "evidence_bytes": evidence_path.stat().st_size,
                    "evidence_sha256": sha256_file(evidence_path),
                    "report_bytes": report_path.stat().st_size,
                    "report_sha256": sha256_file(report_path),
                    "scenario_id": scenario_id,
                }
            )
    if len(validations) != 8:
        raise ValueError("H3 validation did not cover exactly eight scenario/candidate runs")
    norm_summaries = [item["norm_summary_consistency"] for item in validations]
    norm_summary_records = [
        {
            "chunk_size": validation["chunk_size"],
            "scenario_id": validation["scenario_id"],
            **record,
        }
        for validation in validations
        for record in validation["norm_summary_consistency"]["records"]
    ]
    if (
        sum(item["comparisons"] for item in norm_summaries)
        != validator_consistency_amendment["required_summary_comparisons"]
        or len(norm_summary_records) != validator_consistency_amendment["required_summary_comparisons"]
        or {item["element_count"] for item in norm_summaries}
        != {validator_consistency_amendment["bound"]["element_count"]}
        or {item["relative_bound"] for item in norm_summaries}
        != {validator_consistency_amendment["bound"]["upward_rounded_binary64"]}
    ):
        raise ValueError("H3 validator-consistency aggregate accounting drift")

    output = {
        "allowed_conclusion": "All eight R18 H3 runs and their tensor evidence passed independent validation; H4 only is authorized.",
        "allowed_successor": "H4_only",
        "artifact": "qwen35_r18_h3_independent_validation",
        "code_manifest": {"bytes": args.code_manifest.stat().st_size, "sha256": sha256_file(args.code_manifest)},
        "current_validator_commit": current_commit,
        "h2_predecessor": {
            "artifact": h2_validation["artifact"],
            "sha256": sha256_file(args.h2_independent_validation),
            "status": h2_validation["status"],
            "successor_gate_authorized": h2_validation["successor_gate_authorized"],
        },
        "h3_manifest_sha256": h3_digest,
        "harness_amendment": {
            "attempt01_failure_closure_sha256": sha256_file(args.attempt01_failure_closure),
            "human_protocol_sha256": sha256_file(args.harness_amendment_human_protocol),
            "machine_manifest_sha256": harness_amendment_digest,
            "preregistration_closure_sha256": sha256_file(args.harness_amendment_preregistration_closure),
            "status": harness_amendment["status"],
        },
        "human_protocol_sha256": sha256_file(args.human_protocol),
        "preregistration_closure_sha256": sha256_file(args.preregistration_closure),
        "validator_amendment": {
            "amendment_id": validator_amendment["amendment_id"],
            "human_protocol_sha256": sha256_file(args.validator_amendment_human_protocol),
            "machine_manifest_sha256": validator_amendment_digest,
            "preregistration_closure_sha256": sha256_file(args.validator_amendment_preregistration_closure),
            "status": validator_amendment["status"],
        },
        "validator_consistency_amendment": {
            "amendment_id": validator_consistency_amendment["amendment_id"],
            "failed_validator_closure_sha256": sha256_file(args.validator_v2_failure_closure),
            "forensic_closure_sha256": sha256_file(args.validator_failure_forensic_closure),
            "human_protocol_sha256": sha256_file(args.validator_consistency_amendment_human_protocol),
            "machine_manifest_sha256": validator_consistency_amendment_digest,
            "preregistration_closure_sha256": sha256_file(
                args.validator_consistency_amendment_preregistration_closure
            ),
            "status": validator_consistency_amendment["status"],
        },
        "producer_source": producer_source,
        "protocol_id": h3["protocol_id"],
        "report_bindings": report_bindings,
        "norm_summary_consistency_audit": norm_summary_records,
        "schema_version": 1,
        "scientific_training_authorized": False,
        "status": "passed",
        "validation_counts": {
            "active_clipping_paths": sum(item["active_clipping_paths"] for item in validations),
            "aggregate_metric_groups_recomputed": sum(
                item["aggregate_metric_groups_recomputed"] for item in validations
            ),
            "cases": sum(item["cases"] for item in validations),
            "loss_metric_groups_recomputed": sum(item["loss_metric_groups_recomputed"] for item in validations),
            "named_metric_groups_recomputed": sum(item["named_metric_groups_recomputed"] for item in validations),
            "norm_summary_consistency_comparisons": sum(item["comparisons"] for item in norm_summaries),
            "norm_summary_element_count": norm_summaries[0]["element_count"],
            "norm_summary_maximum_absolute_difference": max(
                item["maximum_absolute_difference"] for item in norm_summaries
            ),
            "norm_summary_maximum_binary64_ulp_distance": max(
                item["maximum_binary64_ulp_distance"] for item in norm_summaries
            ),
            "norm_summary_maximum_relative_difference": max(
                item["maximum_relative_difference"] for item in norm_summaries
            ),
            "norm_summary_relative_bound": norm_summaries[0]["relative_bound"],
            "scenario_candidate_runs": len(validations),
            "tensor_keys": sum(item["tensor_keys"] for item in validations),
        },
        "validator_source": {
            "open_instruct/qwen35_qualification_r18_h3.py": sha256_file(
                args.repo_root / "open_instruct/qwen35_qualification_r18_h3.py"
            ),
            "scripts/train/qwen35/validate_qwen35_h3_reports_r18.py": sha256_file(
                args.repo_root / "scripts/train/qwen35/validate_qwen35_h3_reports_r18.py"
            ),
            "scripts/train/qwen35/qwen35_h3_validator_amendment_r18_v1.json": sha256_file(
                args.repo_root / "scripts/train/qwen35/qwen35_h3_validator_amendment_r18_v1.json"
            ),
            "scripts/train/qwen35/qwen35_h3_validator_consistency_amendment_r18_v3.json": sha256_file(
                args.repo_root / "scripts/train/qwen35/qwen35_h3_validator_consistency_amendment_r18_v3.json"
            ),
        },
    }
    write_json_atomic(args.validation_output, output)
    print(json.dumps({"output": str(args.validation_output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
