#!/bin/bash
# Build the separately pinned Leonardo Qwen3.5 environment without running a model.

set -euo pipefail

personal_root=/leonardo_work/AIFAC_F02_434/ytahtah0/fc_causal_v3
if [[ $# -ne 2 ]]; then
    echo "usage: $0 OUTPUT_VENV REQUIREMENTS_FILE" >&2
    exit 2
fi
output_venv=$(realpath -m "$1")
requirements=$(realpath "$2")
case "$output_venv" in
    "$personal_root"/*) ;;
    *) echo "refusing environment outside the personal AIFAC root: $output_venv" >&2; exit 2 ;;
esac
case "$requirements" in
    "$personal_root"/*) ;;
    *) echo "refusing requirements outside the personal AIFAC root: $requirements" >&2; exit 2 ;;
esac
if [[ -e "$output_venv" ]]; then
    echo "refusing pre-existing environment: $output_venv" >&2
    exit 2
fi

module load gcc/12.2.0
module load cuda/12.6
uv=/leonardo/home/userexternal/ytahtah0/.local/bin/uv
python=/leonardo/home/userexternal/ytahtah0/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
"$uv" venv --python "$python" "$output_venv"

# Install the complete top-level contract only after Torch is visible to the
# extension builds. MAX_JOBS bounds login-node load if a matching release wheel
# is unavailable; such a source build is reviewed and stopped rather than left
# unbounded.
"$uv" pip install --no-config --python "$output_venv/bin/python" \
    'setuptools==79.0.1' wheel ninja packaging
: "${CUDA_HOME:?cuda/12.6 module did not set CUDA_HOME}"
export MAX_JOBS=2
# Both indexes are explicit, official inputs to this frozen environment. The
# strategy is necessary because uv otherwise stops at the CUDA index when it
# sees ordinary dependencies such as NumPy there, even when the pinned version
# is available only on PyPI.
grep -Ev '^(flash-attn|causal-conv1d|flash-linear-attention|fla-core)(==|[[:space:]]*@)' \
    "$requirements" >"$output_venv/base-requirements.txt"
"$uv" pip install --no-config --python "$output_venv/bin/python" \
    --index-strategy unsafe-best-match -r "$output_venv/base-requirements.txt"
# A separate transaction is mandatory: flash-attn and causal-conv1d inspect
# the already-installed Torch build from setup.py but do not declare it as a
# PEP 517 build dependency.
export TORCH_CUDA_ARCH_LIST=8.0
export CMAKE_BUILD_PARALLEL_LEVEL=2
# Both upstream setup.py files otherwise wrap bdist_wheel with a command that
# downloads a release wheel even when pip/uv was given an sdist. Force the real
# local C++/CUDA extension path so the resulting ELF ABI is Leonardo-native.
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
native_sdist_dir="$output_venv/pinned-sources/native-sdists"
mkdir -p "$native_sdist_dir"
causal_sdist="$native_sdist_dir/causal_conv1d-1.6.2.post1.tar.gz"
flash_sdist="$native_sdist_dir/flash_attn-2.8.3.tar.gz"
curl --fail --location --silent --show-error \
    'https://files.pythonhosted.org/packages/63/5c/2403b8410122d159405c4bd8456340c7251c193358fa24d30cb273fb5048/causal_conv1d-1.6.2.post1.tar.gz' \
    --output "$causal_sdist"
curl --fail --location --silent --show-error \
    'https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz' \
    --output "$flash_sdist"
printf '%s  %s\n' \
    245e314ea21064ded7a5bf6b3b842b644aa6f92e45cecfe3e935629744c35ff4 "$causal_sdist" \
    1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d "$flash_sdist" \
    | sha256sum --check --strict
grep -Ev '^(flash-attn|causal-conv1d)(==|[[:space:]]*@)' \
    "$requirements" >"$output_venv/non-native-requirements.txt"
"$uv" pip install --no-config --python "$output_venv/bin/python" --no-build-isolation \
    --no-cache --no-deps "$causal_sdist"
"$uv" pip install --no-config --python "$output_venv/bin/python" --no-build-isolation \
    --no-cache --no-deps "$flash_sdist"
"$uv" pip install --no-config --python "$output_venv/bin/python" \
    --index-strategy unsafe-best-match -r "$output_venv/non-native-requirements.txt"

# Liger commit 72a4ed tracks ops/experimental/*.py without an __init__.py while
# declaring namespaces=false in setuptools package discovery. Its wheel thus
# omits modules that ops/__init__.py imports. Preserve the exact pinned runtime
# code: keep the archive-installed distribution/metadata, fetch a clean detached
# checkout of the same commit, and make its src tree precede site-packages.
liger_commit=72a4ed47a5c593b58045a0af14d3f774a037bd92
liger_checkout="$output_venv/pinned-sources/liger-kernel"
mkdir -p "$(dirname "$liger_checkout")"
git init "$liger_checkout"
git -C "$liger_checkout" remote add origin https://github.com/linkedin/Liger-Kernel.git
git -C "$liger_checkout" fetch --depth=1 origin "$liger_commit"
git -C "$liger_checkout" checkout --detach FETCH_HEAD
if [[ "$(git -C "$liger_checkout" rev-parse HEAD)" != "$liger_commit" || \
      -n "$(git -C "$liger_checkout" status --porcelain=v1)" ]]; then
    echo "pinned Liger source checkout failed identity/cleanliness validation" >&2
    exit 2
fi
site_packages=$("$output_venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$output_venv/bin/python" -c \
    'import pathlib,sys; pathlib.Path(sys.argv[2]).write_text(f"import sys; sys.path.insert(0, {sys.argv[1]!r})\n")' \
    "$liger_checkout/src" "$site_packages/00-qwen35-pinned-liger-source.pth"

"$uv" pip check --python "$output_venv/bin/python"
"$uv" pip freeze --python "$output_venv/bin/python" | LC_ALL=C sort >"$output_venv/pip-freeze.txt"
"$output_venv/bin/python" - <<'PY' >"$output_venv/runtime-import-report.json"
import importlib.metadata as metadata
import importlib
import hashlib
import json
import platform
import pathlib
import re
import subprocess
import sys
import urllib.parse

packages = (
    "accelerate",
    "causal-conv1d",
    "flash-attn",
    "flash-linear-attention",
    "fla-core",
    "liger-kernel",
    "numpy",
    "torch",
    "torchvision",
    "transformers",
)
versions = {package: metadata.version(package) for package in packages}
expected_source_commits = {
    "transformers": "d7d894cf917562d62c61497588ab64e4ae2c699d",
    "liger-kernel": "72a4ed47a5c593b58045a0af14d3f774a037bd92",
}
source_pins = {}
for package, expected_commit in expected_source_commits.items():
    direct_url_text = metadata.distribution(package).read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(f"{package} has no direct_url.json")
    direct_url = json.loads(direct_url_text)
    source_url = str(direct_url.get("url", ""))
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    archive_pinned = expected_commit in source_url and "/archive/" in source_url
    if installed_commit != expected_commit and not archive_pinned:
        raise RuntimeError(f"{package} source pin mismatch: {installed_commit!r} / {source_url!r}")
    source_pins[package] = {"expected_commit": expected_commit, "source_url": source_url}
native_sources = {
    "causal-conv1d": {
        "module": "causal_conv1d_cuda",
        "sdist_url": "https://files.pythonhosted.org/packages/63/5c/2403b8410122d159405c4bd8456340c7251c193358fa24d30cb273fb5048/causal_conv1d-1.6.2.post1.tar.gz",
        "sdist_sha256": "245e314ea21064ded7a5bf6b3b842b644aa6f92e45cecfe3e935629744c35ff4",
        "sdist_filename": "causal_conv1d-1.6.2.post1.tar.gz",
        "build_mode": "forced_local_source_build_including_sm80_no_cache",
    },
    "flash-attn": {
        "module": "flash_attn_2_cuda",
        "sdist_url": "https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz",
        "sdist_sha256": "1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d",
        "sdist_filename": "flash_attn-2.8.3.tar.gz",
        "build_mode": "forced_local_source_build_sm80_only_no_cache",
    },
}
for package, expected in native_sources.items():
    direct_url_text = metadata.distribution(package).read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(f"{package} source build has no direct_url.json")
    direct_url = json.loads(direct_url_text)
    observed_url = str(direct_url.get("url", ""))
    parsed_url = urllib.parse.urlparse(observed_url)
    observed_path = pathlib.Path(urllib.parse.unquote(parsed_url.path)).resolve()
    expected_path = (pathlib.Path(sys.prefix) / "pinned-sources/native-sdists" / expected["sdist_filename"]).resolve()
    if parsed_url.scheme != "file" or observed_path != expected_path:
        raise RuntimeError(f"{package} installed-source path drift: {observed_url!r}")
    observed_sdist_hash = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    if observed_sdist_hash != expected["sdist_sha256"]:
        raise RuntimeError(f"{package} preserved sdist hash drift: {observed_sdist_hash}")
    source_pins[package] = {
        "source_url": expected["sdist_url"],
        "installed_from": observed_url,
        "sdist_path": str(expected_path),
        "sdist_sha256": expected["sdist_sha256"],
        "build_mode": expected["build_mode"],
    }
import torch
import transformers
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
import liger_kernel
from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

native_extension_abi = {}
host_glibc = tuple(int(part) for part in platform.libc_ver()[1].split(".")[:2])
if host_glibc != (2, 28):
    raise RuntimeError(f"unexpected Leonardo build-host glibc: {host_glibc}")
for package, expected in native_sources.items():
    module = importlib.import_module(expected["module"])
    path = pathlib.Path(module.__file__).resolve()
    readelf = subprocess.run(
        ["readelf", "--version-info", str(path)], check=True, text=True, capture_output=True
    ).stdout
    glibc_versions = sorted({tuple(map(int, match)) for match in re.findall(r"GLIBC_(\d+)\.(\d+)", readelf)})
    maximum_glibc = max(glibc_versions, default=(0, 0))
    if maximum_glibc > host_glibc:
        raise RuntimeError(f"{package} requires GLIBC_{maximum_glibc}, host is {host_glibc}")
    native_extension_abi[package] = {
        "module": expected["module"],
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "required_glibc_versions": [list(value) for value in glibc_versions],
        "maximum_glibc_version": list(maximum_glibc),
        "host_glibc_version": list(host_glibc),
        "import_status": "passed",
    }

report = {
    "artifact": "qwen35_leonardo_cpu_import_preflight",
    "schema_version": 1,
    "status": "passed",
    "python": sys.version,
    "packages": versions,
    "source_pins": source_pins,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available_on_login": torch.cuda.is_available(),
    "qwen35_conditional_class": Qwen3_5ForConditionalGeneration.__name__,
    "qwen35_text_class": Qwen3_5ForCausalLM.__name__,
    "liger_qwen35_patch": apply_liger_kernel_to_qwen3_5.__name__,
    "transformers_version": transformers.__version__,
    "native_extension_abi": native_extension_abi,
}
liger_checkout = pathlib.Path(sys.prefix) / "pinned-sources" / "liger-kernel"
liger_head = subprocess.run(
    ["git", "-C", str(liger_checkout), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
).stdout.strip()
liger_status = subprocess.run(
    ["git", "-C", str(liger_checkout), "status", "--porcelain=v1"], check=True, text=True, capture_output=True
).stdout
liger_import_path = pathlib.Path(liger_kernel.__file__).resolve()
if liger_head != expected_source_commits["liger-kernel"] or liger_status:
    raise RuntimeError("pinned Liger checkout identity or cleanliness drift")
if not liger_import_path.is_relative_to((liger_checkout / "src").resolve()):
    raise RuntimeError(f"Liger did not import from the pinned source checkout: {liger_import_path}")
report["source_pins"]["liger-kernel"].update(
    {
        "runtime_import_mode": "clean_detached_pinned_source_checkout_precedes_installed_distribution",
        "checkout_path": str(liger_checkout.resolve()),
        "checkout_head": liger_head,
        "checkout_clean": True,
        "runtime_import_path": str(liger_import_path),
    }
)
print(json.dumps(report, indent=2, sort_keys=True))
PY
date --iso-8601=seconds >"$output_venv/.environment_complete"
