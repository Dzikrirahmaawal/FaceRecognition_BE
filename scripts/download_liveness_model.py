#!/usr/bin/env python3
"""Download and verify the MiniFASNet V2 liveness model.

Downloads from the pinned Hugging Face URL, verifies size and SHA-256,
and writes atomically (temp file then rename). Skips download if the
existing file already passes the checksum. Never overwrites a valid model.

Model: MiniFASNet V2 (2.7_80x80) — Apache 2.0
Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
ONNX: https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from urllib.request import urlretrieve

# Pinned model metadata
MODEL_URL = (
    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx"
    "/resolve/d29c87568ca9b5662da803b10f217c4db20b142b/minifasnet_v2.onnx"
)
MODEL_FILENAME = "minifasnet_v2.onnx"
MODEL_SIZE_BYTES = 1_744_116
MODEL_SHA256 = "d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b"

LICENSE_NOTE = (
    "Model weights are licensed under Apache 2.0 (same as upstream "
    "minivision-ai/Silent-Face-Anti-Spoofing). "
    "See the model card on Hugging Face for license details."
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_valid(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) != MODEL_SIZE_BYTES:
        return False
    return _sha256_file(path) == MODEL_SHA256


def main() -> int:
    models_dir = os.environ.get("FACE_MODELS_DIR", "models")
    dest = os.path.join(models_dir, MODEL_FILENAME)

    if _model_valid(dest):
        print(f"Skipping {MODEL_FILENAME}: already present and verified at {dest}")
        return 0

    os.makedirs(models_dir, exist_ok=True)

    print(f"Downloading {MODEL_FILENAME} from {MODEL_URL}")
    print(f"License: {LICENSE_NOTE}")

    fd, tmp_path = tempfile.mkstemp(dir=models_dir, suffix=".tmp")
    try:
        urlretrieve(MODEL_URL, tmp_path)
        actual_size = os.path.getsize(tmp_path)
        actual_sha = _sha256_file(tmp_path)

        if actual_size != MODEL_SIZE_BYTES:
            print(
                f"ERROR: Downloaded size {actual_size} != expected {MODEL_SIZE_BYTES}",
                file=sys.stderr,
            )
            return 1
        if actual_sha != MODEL_SHA256:
            print(
                f"ERROR: SHA-256 mismatch: {actual_sha} != {MODEL_SHA256}",
                file=sys.stderr,
            )
            return 1

        os.replace(tmp_path, dest)
        print(f"Saved {MODEL_FILENAME} to {dest} (verified SHA-256)")
        return 0
    except Exception as exc:
        print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return 1
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())