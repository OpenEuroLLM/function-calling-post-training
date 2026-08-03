#!/usr/bin/env python3
"""Capture and validate immutable H0 identity/account/runtime/hardware evidence."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import urllib.parse
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from open_instruct.qwen35_qualification import parse_glibc_versions, sha256_file
from open_instruct.qwen35_qualification_loader import load_qualification_manifest
from open_instruct.qwen35_training import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity-file", type=Path, action="append", default=[])
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None, required: bool = True) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)  # noqa: S603
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if required and completed.returncode != 0:
        raise RuntimeError(f"inventory command failed: {result}")
    return result


def package_identity(name: str) -> dict[str, Any]:
    distribution = metadata.distribution(name)
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else None
    return {"version": distribution.version, "direct_url": direct_url}


def pinned_package_identity(name: str, expected_commit: str) -> dict[str, Any]:
    identity = package_identity(name)
    direct_url = identity.get("direct_url") or {}
    source_url = str(direct_url.get("url", ""))
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    if installed_commit != expected_commit and expected_commit not in source_url:
        raise RuntimeError(f"{name} source commit drift: {installed_commit!r} / {source_url!r}")
    identity["expected_commit"] = expected_commit
    return identity


def native_extension_identity(package: str, specification: dict[str, Any]) -> dict[str, Any]:
    identity = package_identity(package)
    direct_url = identity.get("direct_url") or {}
    observed_url = str(direct_url.get("url", ""))
    parsed_url = urllib.parse.urlparse(observed_url)
    observed_path = Path(urllib.parse.unquote(parsed_url.path)).resolve()
    filename = Path(urllib.parse.urlparse(specification["sdist_url"]).path).name
    expected_path = (Path(sys.prefix) / "pinned-sources/native-sdists" / filename).resolve()
    if parsed_url.scheme != "file" or observed_path != expected_path:
        raise RuntimeError(f"{package} installed-source path drift: {observed_url!r}")
    expected_hash = specification["sdist_sha256"]
    observed_sdist_hash = sha256_file(expected_path)
    if observed_sdist_hash != expected_hash:
        raise RuntimeError(f"{package} preserved sdist hash drift: {observed_sdist_hash}")
    module = importlib.import_module(specification["module"])
    path = Path(module.__file__).resolve()
    readelf = run(["readelf", "--version-info", str(path)])
    glibc_versions = parse_glibc_versions(readelf["stdout"])
    maximum_glibc = max(glibc_versions, default=[0, 0])
    if maximum_glibc > specification["maximum_glibc_version"]:
        raise RuntimeError(f"{package} GLIBC ABI drift: {maximum_glibc} > {specification['maximum_glibc_version']}")
    identity.update(
        {
            "module": specification["module"],
            "official_sdist_url": specification["sdist_url"],
            "preserved_sdist_path": str(expected_path),
            "preserved_sdist_sha256": observed_sdist_hash,
            "path": str(path),
            "sha256": sha256_file(path),
            "required_glibc_versions": glibc_versions,
            "maximum_glibc_version": maximum_glibc,
            "build_mode": specification["build_mode"],
            "readelf": readelf,
            "import_status": "passed",
        }
    )
    return identity


def nccl_version() -> list[int] | None:
    try:
        return list(torch.cuda.nccl.version())
    except (AttributeError, RuntimeError):
        return None


def expected_runtime_versions(runtime_pins: dict[str, Any]) -> dict[str, str]:
    versions = {
        "accelerate": runtime_pins["accelerate_version"],
        "causal-conv1d": runtime_pins["causal_conv1d_version"],
        "flash-attn": runtime_pins["flash_attn_version"],
        "flash-linear-attention": runtime_pins["flash_linear_attention_version"],
        "fla-core": runtime_pins["fla_core_version"],
        "numpy": runtime_pins["numpy_version"],
        "torch": runtime_pins["torch_version"],
        "torchvision": runtime_pins["torchvision_version"],
        "transformers": runtime_pins["transformers_version"],
    }
    if "liger_version" in runtime_pins:
        versions["liger-kernel"] = runtime_pins["liger_version"]
    elif runtime_pins.get("liger_execution_allowed") is not False:
        raise RuntimeError("a runtime without a pinned Liger identity must explicitly forbid Liger execution")
    return versions


def liger_runtime_identity(runtime_pins: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if "liger_version" not in runtime_pins:
        imported = any(name == "liger_kernel" or name.startswith("liger_kernel.") for name in sys.modules)
        if imported:
            raise RuntimeError("the non-Liger H0 process imported Liger")
        return (
            {
                "execution_allowed": False,
                "imported": False,
                "installed_distribution_version": metadata.version("liger-kernel"),
            },
            None,
        )

    liger_kernel = importlib.import_module("liger_kernel")
    liger_checkout = Path(sys.prefix) / "pinned-sources" / "liger-kernel"
    liger_head = run(["git", "-C", str(liger_checkout), "rev-parse", "HEAD"])["stdout"].strip()
    liger_status = run(["git", "-C", str(liger_checkout), "status", "--porcelain=v1"])["stdout"]
    liger_import_path = Path(liger_kernel.__file__).resolve()
    if liger_head != runtime_pins["liger_commit"] or liger_status:
        raise RuntimeError("pinned Liger runtime checkout identity or cleanliness drift")
    if not liger_import_path.is_relative_to((liger_checkout / "src").resolve()):
        raise RuntimeError(f"Liger import path does not use the pinned source checkout: {liger_import_path}")
    return (
        pinned_package_identity("liger-kernel", runtime_pins["liger_commit"]),
        {
            "import_mode": runtime_pins["liger_import_mode"],
            "checkout_path": str(liger_checkout.resolve()),
            "checkout_head": liger_head,
            "checkout_clean": True,
            "runtime_import_path": str(liger_import_path),
        },
    )


def main() -> None:
    args = parse_args()
    qualification, qualification_sha256 = load_qualification_manifest(args.qualification_manifest)
    runtime_pins = qualification["runtime_pins"]
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    if account != qualification["scope"]["slurm_account"]:
        raise RuntimeError(f"Slurm account mismatch: {account!r}")
    output = args.report_output.resolve()
    personal_root = Path(qualification["scope"]["personal_output_root"])
    if not output.is_relative_to(personal_root):
        raise RuntimeError(f"inventory output {output} is outside personal root {personal_root}")
    repo = args.repo_root.resolve()
    head = run(["git", "rev-parse", "HEAD"], cwd=repo)["stdout"].strip()
    expected_head = os.environ.get("QWEN35_EXPECTED_CODE_COMMIT")
    if not expected_head or head != expected_head:
        raise RuntimeError(f"qualification source commit drift: {head!r} != {expected_head!r}")
    status = run(["git", "status", "--porcelain=v1"], cwd=repo)["stdout"]
    if status:
        raise RuntimeError("qualification source worktree is dirty")
    baseline = qualification["source"]["corrective_baseline_commit"]
    ancestry = run(["git", "merge-base", "--is-ancestor", baseline, head], cwd=repo, required=False)
    if ancestry["returncode"] != 0:
        raise RuntimeError(f"qualification source {head} is not descended from corrective baseline {baseline}")
    if not torch.cuda.is_available():
        raise RuntimeError("H0 hardware inventory requires CUDA")

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("H0 hardware inventory must execute inside a Slurm allocation")
    slurm_job = run(["scontrol", "show", "job", "-o", slurm_job_id])
    if f"Account={account}" not in slurm_job["stdout"]:
        raise RuntimeError("scontrol job record does not confirm the personal account")

    identities = {}
    for path in args.identity_file:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        identities[str(resolved)] = {"size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}

    expected_versions = expected_runtime_versions(runtime_pins)
    observed_versions = {name: metadata.version(name) for name in expected_versions}
    if observed_versions != expected_versions or torch.version.cuda != runtime_pins["torch_cuda_build"]:
        raise RuntimeError("qualification runtime version drift")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != runtime_pins["python_major_minor"]:
        raise RuntimeError("qualification Python version drift")
    liger_identity, liger_checkout_identity = liger_runtime_identity(runtime_pins)

    native_extensions = {
        package: native_extension_identity(package, specification)
        for package, specification in runtime_pins["native_extensions"].items()
    }

    hardware_acceptance = qualification["hardware_acceptance"]
    gpu_properties = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_properties.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "major": properties.major,
                "minor": properties.minor,
                "multi_processor_count": properties.multi_processor_count,
            }
        )
        if hardware_acceptance["gpu_name_contains"] not in properties.name:
            raise RuntimeError(f"ineligible GPU for R3: {properties.name}")
        if properties.total_memory < int(hardware_acceptance["minimum_device_memory_bytes"]):
            raise RuntimeError(f"GPU memory below the R1 floor: {properties.total_memory}")
        if [properties.major, properties.minor] != hardware_acceptance["compute_capability"]:
            raise RuntimeError(f"GPU compute-capability drift: {(properties.major, properties.minor)}")

    observed_nccl_version = nccl_version()
    if observed_nccl_version is None:
        raise RuntimeError("the pinned Torch runtime exposes no NCCL version")
    nvidia_smi_query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version,pstate,clocks.sm,clocks.mem,ecc.errors.uncorrected.aggregate.total,mig.mode.current",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in nvidia_smi_query["stdout"].splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 10:
            raise RuntimeError(f"unexpected nvidia-smi qualification row: {line!r}")
        if fields[8] != "0":
            raise RuntimeError(f"GPU {fields[0]} has aggregate uncorrected ECC errors: {fields[8]!r}")
        if fields[9].lower() != "disabled":
            raise RuntimeError(f"GPU {fields[0]} has unexpected MIG mode: {fields[9]!r}")

    report = {
        "artifact": "qwen35_h0_hardware_inventory",
        "schema_version": 1,
        "status": "passed",
        "qualification_protocol_id": qualification["protocol_id"],
        "qualification_manifest_sha256": qualification_sha256,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
        "slurm": {
            "job_id": slurm_job_id,
            "job_account": account,
            "job_record": slurm_job,
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
            "gpus": os.environ.get("SLURM_GPUS"),
        },
        "source": {"repo": str(repo), "head": head, "baseline": baseline, "clean": True},
        "runtime": {
            "packages": observed_versions,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": observed_nccl_version,
            "triton": package_identity("triton"),
            "transformers": pinned_package_identity("transformers", runtime_pins["transformers_commit"]),
            "accelerate": package_identity("accelerate"),
            "liger_kernel": liger_identity,
            "liger_runtime_checkout": liger_checkout_identity,
            "native_extensions": native_extensions,
        },
        "gpu_properties": gpu_properties,
        "nvidia_smi_query": nvidia_smi_query,
        "nvidia_smi_topology": run(["nvidia-smi", "topo", "-m"]),
        "nvidia_smi_query_full": run(["nvidia-smi", "-q"]),
        "identity_files": identities,
    }
    write_json_atomic(output, report)
    print(json.dumps({"output": str(output), "status": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
