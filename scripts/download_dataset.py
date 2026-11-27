"""Download the public Roboflow dataset "brazilian_lpr" in YOLO format.

Source: https://universe.roboflow.com/lpr-rn1pv/brazilian_lpr
License: CC BY 4.0 - attribution to the dataset authors (workspace
``lpr-rn1pv`` on Roboflow Universe) is required when images are redistributed.

Why not the usual "export a version"?
-------------------------------------
Roboflow only exports *versions* of a project, and this project has none
published (``GET /lpr-rn1pv/brazilian_lpr`` returns ``"versions": []``). The
script therefore uses the image API instead: it lists every image with
``POST /{workspace}/{project}/search``, reads each image's annotation with
``GET /{workspace}/{project}/images/{id}`` and downloads the original file.
The train / valid / test split chosen by the dataset authors is kept.

Every plate class of the dataset (``PLATE``, ``plate``, ``placa carro``,
``placa moto``, ``placa carro mercosul``, ``placa moto mercosul``) becomes the
single YOLO class ``plate``; ``CAR`` boxes are ignored. The vehicle type
(``car`` / ``motorcycle`` / ``unknown``) is kept in ``manifest.json`` so the
evaluation can report each group separately.

Output (outside git, ``datasets/`` is ignored)::

    datasets/brazilian_lpr/
        data.yaml                    # Ultralytics dataset file
        manifest.json                # one entry per image: split, boxes, labels, vehicle type
        {train,valid,test}/images/*.jpg
        {train,valid,test}/labels/*.txt   # YOLO: class cx cy w h (normalized)

Usage (from the project root, API key in the environment, never in the code)::

    set ROBOFLOW_API_KEY=...        (PowerShell: $env:ROBOFLOW_API_KEY="...")
    python -m scripts.download_dataset
    python -m scripts.download_dataset --force    # download again
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

API = "https://api.roboflow.com"
WORKSPACE = "lpr-rn1pv"
PROJECT = "brazilian_lpr"
DATASET_URL = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}"
LICENSE = "CC BY 4.0"
DATASET_DIR = config.BASE_DIR / "datasets" / PROJECT
SPLITS = ("train", "valid", "test")
PAGE_SIZE = 100
WORKERS = 8
RETRIES = 3
TIMEOUT = 60

ENV_KEY = "ROBOFLOW_API_KEY"


@dataclass
class PlateBox:
    """Plate box in pixels: center x/y, width, height (Roboflow convention)."""

    cx: float
    cy: float
    w: float
    h: float
    label: str

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (round(self.cx - self.w / 2), round(self.cy - self.h / 2),
                round(self.cx + self.w / 2), round(self.cy + self.h / 2))


@dataclass
class DatasetImage:
    id: str
    name: str                  # original file name on Roboflow
    file: str                  # path relative to the dataset folder
    split: str
    width: int
    height: int
    vehicle: str               # "car", "motorcycle" or "unknown"
    plates: list[PlateBox] = field(default_factory=list)


# --------------------------------------------------------------------------
# Annotation conversion (pure, unit tested)
# --------------------------------------------------------------------------
def is_plate_label(label: str) -> bool:
    text = label.strip().lower()
    return "plate" in text or "placa" in text


def vehicle_type(labels: list[str]) -> str:
    """Vehicle type from the class names of one image."""
    lowered = [label.lower() for label in labels]
    if any("moto" in label for label in lowered):
        return "motorcycle"
    if any("carro" in label for label in lowered) or "car" in lowered:
        return "car"
    return "unknown"


def parse_box(raw: dict) -> PlateBox:
    """Roboflow box (strings, centre format; polygons carry ``points``) -> :class:`PlateBox`."""
    if raw.get("points"):
        xs = [p[0] for p in raw["points"]]
        ys = [p[1] for p in raw["points"]]
        return PlateBox((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys),
                        raw.get("label", ""))
    return PlateBox(float(raw["x"]), float(raw["y"]), float(raw["width"]), float(raw["height"]),
                    raw.get("label", ""))


def yolo_lines(plates: list[PlateBox], width: int, height: int) -> list[str]:
    """YOLO label lines (class 0), coordinates normalized and clipped to [0, 1]."""
    lines = []
    for box in plates:
        x1, y1 = max(0.0, box.cx - box.w / 2), max(0.0, box.cy - box.h / 2)
        x2, y2 = min(float(width), box.cx + box.w / 2), min(float(height), box.cy + box.h / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def safe_name(image_id: str, name: str) -> str:
    """Unique file name: Roboflow names may repeat between images."""
    stem = Path(name).stem.replace(" ", "_")
    return f"{stem}_{image_id[:8]}.jpg"


def data_yaml(root: Path) -> str:
    return (f"# {PROJECT} ({LICENSE}) - {DATASET_URL}\n"
            f"path: {root.as_posix()}\n"
            "train: train/images\nval: valid/images\ntest: test/images\n"
            "names:\n  0: plate\n")


# --------------------------------------------------------------------------
# Roboflow API
# --------------------------------------------------------------------------
def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.request(method, url, timeout=TIMEOUT, **kwargs)
            if response.status_code < 500:
                response.raise_for_status()
                return response
        except requests.ConnectionError:
            if attempt == RETRIES:
                raise
        time.sleep(2 * attempt)
    response.raise_for_status()
    return response


def list_images(session: requests.Session, key: str) -> list[dict]:
    """Every image of the project (id, name, split), following the pagination."""
    items: list[dict] = []
    while True:
        response = _request(session, "POST", f"{API}/{WORKSPACE}/{PROJECT}/search", params={"api_key": key},
                            json={"offset": len(items), "limit": PAGE_SIZE, "fields": ["id", "name", "split"]})
        data = response.json()
        items += data["results"]
        if not data["results"] or len(items) >= data["total"]:
            return items


def download_image(session: requests.Session, key: str, item: dict, root: Path, force: bool) -> DatasetImage | None:
    """Fetch annotation + original image; writes the image and its YOLO label."""
    split = item.get("split") if item.get("split") in SPLITS else "train"
    detail = _request(session, "GET", f"{API}/{WORKSPACE}/{PROJECT}/images/{item['id']}",
                      params={"api_key": key}).json()["image"]
    annotation = detail.get("annotation") or {}
    boxes = annotation.get("boxes") or []
    labels = [b.get("label", "") for b in boxes]
    plates = [parse_box(b) for b in boxes if is_plate_label(b.get("label", ""))]
    width, height = int(annotation.get("width", 0)), int(annotation.get("height", 0))
    if not width or not height:
        return None
    name = safe_name(item["id"], item.get("name", item["id"]))
    image_path = root / split / "images" / name
    if force or not image_path.is_file():
        content = _request(session, "GET", detail["urls"]["original"]).content
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(content)
    label_path = root / split / "labels" / f"{Path(name).stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(yolo_lines(plates, width, height)) + "\n", encoding="utf-8")
    return DatasetImage(item["id"], item.get("name", ""), image_path.relative_to(root).as_posix(), split, width, height,
                        vehicle_type(labels), plates)


def download(force: bool = False, root: Path = DATASET_DIR) -> Path:
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        raise SystemExit(f"Set the {ENV_KEY} environment variable (Roboflow > Settings > API Keys).")
    root.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        items = list_images(session, key)
        print(f"{len(items)} images listed in {WORKSPACE}/{PROJECT}. Downloading...")
        images: list[DatasetImage] = []
        failures: list[str] = []
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(download_image, session, key, item, root, force): item for item in items}
            for done, future in enumerate(as_completed(futures), start=1):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:   # report and continue with the other images
                    failures.append(f"{item.get('name')}: {exc.__class__.__name__}")
                    continue
                if result is None:
                    failures.append(f"{item.get('name')}: missing annotation size")
                else:
                    images.append(result)
                if done % 50 == 0:
                    print(f"  {done}/{len(items)}")
    images.sort(key=lambda i: i.file)
    manifest = {"source": DATASET_URL, "license": LICENSE, "images": [asdict(i) for i in images]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (root / "data.yaml").write_text(data_yaml(root), encoding="utf-8")
    summary = {split: sum(i.split == split for i in images) for split in SPLITS}
    print(f"Saved {len(images)} images to {root} {summary}.")
    if failures:
        print(f"{len(failures)} image(s) failed:\n  " + "\n  ".join(failures))
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download the images again")
    download(parser.parse_args().force)
