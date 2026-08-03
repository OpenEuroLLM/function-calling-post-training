"""Dispatch hash-bound Qwen3.5 qualification overlays by protocol identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from open_instruct.qwen35_qualification_r16 import load_qualification_manifest as load_r16_manifest
from open_instruct.qwen35_qualification_r17 import load_qualification_manifest as load_r17_manifest
from open_instruct.qwen35_qualification_r18 import load_qualification_manifest as load_r18_manifest


def load_qualification_manifest(path: Path) -> tuple[dict[str, Any], str]:
    protocol_id = json.loads(path.read_text()).get("protocol_id")
    if protocol_id == "qwen35-hardware-qualification-r16":
        loader = load_r16_manifest
    elif protocol_id == "qwen35-hardware-qualification-r17":
        loader = load_r17_manifest
    elif protocol_id == "qwen35-hardware-qualification-r18":
        loader = load_r18_manifest
    else:
        raise ValueError(f"unsupported Qwen3.5 qualification overlay protocol: {protocol_id!r}")
    value, digest = loader(path)
    if value.get("protocol_id") != protocol_id:
        raise ValueError("qualification dispatcher/validated protocol disagreement")
    return value, digest
