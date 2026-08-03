import json

import pytest
from scripts.data.shard_verified_transfer import reassemble_file, split_file


def test_verified_sharded_transfer_round_trip(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(251)) * 1000)
    shard_dir = tmp_path / "shards"

    split_file(source, shard_dir, chunk_size=32768, overwrite=False)
    manifest = json.loads((shard_dir / "transfer_manifest.json").read_text())
    output = tmp_path / "output.bin"
    reassemble_file(shard_dir / "transfer_manifest.json", shard_dir, output, overwrite=False)

    assert len(manifest["shards"]) == 8
    assert output.read_bytes() == source.read_bytes()
    assert manifest["size_bytes"] == source.stat().st_size


def test_verified_sharded_transfer_rejects_corrupt_shard(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefgh" * 100)
    shard_dir = tmp_path / "shards"
    split_file(source, shard_dir, chunk_size=128, overwrite=False)
    manifest = json.loads((shard_dir / "transfer_manifest.json").read_text())
    corrupt = shard_dir / manifest["shards"][2]["path"]
    payload = bytearray(corrupt.read_bytes())
    payload[0] ^= 1
    corrupt.write_bytes(payload)

    with pytest.raises(ValueError, match="shard SHA-256 drift"):
        reassemble_file(shard_dir / "transfer_manifest.json", shard_dir, tmp_path / "output.bin", overwrite=False)
    assert not (tmp_path / "output.bin").exists()
