"""Download the pre-trained YOLO11 license plate detector.

Source: https://huggingface.co/morsetechlab/yolov11-license-plate-detection
License: AGPL-3.0 (same as Ultralytics). Trained on the Roboflow
"License Plate Recognition" dataset (~10k images, multiple countries).

The download is pinned to an exact repository revision and verified with
SHA-256, so every installation uses byte-identical weights. ``.pt`` files are
Python pickles; pinning + hashing guarantees we only load the file that was
reviewed.

Usage (from the project root)::

    python -m scripts.download_model
    python -m scripts.download_model --force   # re-download
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

REPO_ID = "morsetechlab/yolov11-license-plate-detection"
REVISION = "251a30d7daedca065f56e04b0af04052c907c68f"
FILENAME = "license-plate-finetune-v1s.pt"
SHA256 = "95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> Path:
    """Download (if needed) and verify the weights. Returns the local path."""
    target = config.PUBLISHED_MODEL_PATH
    config.ensure_directories()
    if target.exists() and not force:
        if sha256_of(target) == SHA256:
            print(f"Model already present and verified: {target}")
            return target
        print("Existing file has an unexpected hash; downloading again.")

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download

    print(f"Downloading {FILENAME} from {REPO_ID}@{REVISION[:8]} ...")
    cached = Path(hf_hub_download(REPO_ID, FILENAME, revision=REVISION))
    actual = sha256_of(cached)
    if actual != SHA256:
        raise RuntimeError(f"SHA-256 mismatch: expected {SHA256}, got {actual}. File not installed.")
    shutil.copyfile(cached, target)
    print(f"Saved and verified: {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    download(parser.parse_args().force)
