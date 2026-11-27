"""Evaluate plate detection and OCR on the real Brazilian dataset.

Run after ``python -m scripts.download_dataset``::

    python -m scripts.evaluate                       # every split, published weights
    python -m scripts.evaluate --split test          # one split
    python -m scripts.evaluate --model runs/.../best.pt --tag finetuned

What is measured
----------------
**Detection** - the real :class:`modules.detection.PlateDetector` (YOLO ->
YOLO on CLAHE-normalized image -> classical OpenCV fallback) runs on every
image. A ground-truth plate counts as *detected* when a predicted box overlaps
it with IoU >= 0.5 (greedy one-to-one matching, most confident box first).
Reported: recall (detection rate), precision, mean IoU of the matches, split by
vehicle type and by the method that produced the box. Because the classical
fallback only runs when YOLO finds nothing, the "YOLO only" numbers are exact.

**OCR** - the dataset has bounding boxes but **no plate text**, so OCR
accuracy cannot be computed from it. Without ground truth the script reports
only what is observable: how many reads produced a valid Brazilian plate and
how many would go to manual review. OCR runs on two crops: the box chosen by
the detector (what the system really reads) and the ground-truth box (OCR with
a perfect detector). If ``<dataset>/plate_texts.csv`` (``file;plate``) is
filled in by a person, plate-level and character-level accuracy are computed
for the annotated images. ``--export-crops`` writes the ground-truth crops and
an empty template to make that annotation easier.

The report (Markdown + JSON with one row per image, errors included) is written
to ``docs/evaluation/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from modules.detection import BBox, PlateDetector, crop_with_padding, iou  # noqa: E402
from modules.image_processing import preprocess_plate  # noqa: E402
from modules.ocr import OCRResult, PlateOCR  # noqa: E402
from modules.plate_format import clean_plate_text  # noqa: E402
from scripts.download_dataset import DATASET_DIR, DATASET_URL, LICENSE  # noqa: E402

IOU_THRESHOLD = 0.5
REPORT_DIR = config.BASE_DIR / "docs" / "evaluation"
TEXTS_FILE = "plate_texts.csv"
VEHICLES = ("car", "motorcycle", "unknown")


# --------------------------------------------------------------------------
# Metrics (pure, unit tested)
# --------------------------------------------------------------------------
def match_boxes(gt: list[BBox], predicted: list[BBox], threshold: float = IOU_THRESHOLD
                ) -> tuple[dict[int, tuple[int, float]], list[int]]:
    """Greedy one-to-one matching.

    ``predicted`` must be sorted by confidence (best first). Returns
    ``{gt_index: (pred_index, iou)}`` and the indices of unmatched predictions
    (false positives).
    """
    matches: dict[int, tuple[int, float]] = {}
    false_positives: list[int] = []
    for p, box in enumerate(predicted):
        candidates = [(iou(box, g), g_index) for g_index, g in enumerate(gt) if g_index not in matches]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= threshold:
            matches[best[1]] = (p, best[0])
        else:
            false_positives.append(p)
    return matches, false_positives


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def char_accuracy(predicted: str | None, truth: str) -> float:
    """1 - edit distance / length of the truth, clipped to [0, 1]. No reading = 0."""
    if not predicted:
        return 0.0
    return max(0.0, 1.0 - levenshtein(predicted, truth) / len(truth))


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@dataclass
class OCRRow:
    plate: str | None
    raw_text: str
    confidence: float
    needs_review: bool

    @classmethod
    def from_result(cls, result: OCRResult | None) -> OCRRow | None:
        if result is None:
            return None
        return cls(result.plate, result.raw_text, round(result.confidence, 4), result.needs_review)


@dataclass
class ImageRow:
    file: str
    split: str
    vehicle: str
    gt_plates: int
    predicted: int
    method: str
    matched: int
    false_positives: int
    ious: list[float] = field(default_factory=list)
    ocr_detected: OCRRow | None = None      # OCR of the box the system would read
    ocr_ground_truth: OCRRow | None = None  # OCR of the first ground-truth box
    truth: str | None = None                # plate text from plate_texts.csv, if any
    error: str | None = None


def gt_boxes(entry: dict) -> list[BBox]:
    return [(round(p["cx"] - p["w"] / 2), round(p["cy"] - p["h"] / 2),
             round(p["cx"] + p["w"] / 2), round(p["cy"] + p["h"] / 2)) for p in entry["plates"]]


def read_ocr(ocr: PlateOCR, crop: np.ndarray) -> OCRResult | None:
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 2:
        return None
    processed = preprocess_plate(crop)
    return ocr.recognize({"enhanced": processed.enhanced, "binary": processed.binary})


def load_texts(root: Path) -> dict[str, str]:
    """``file -> plate`` from the human-made CSV (empty lines ignored)."""
    path = root / TEXTS_FILE
    if not path.is_file():
        return {}
    texts = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            plate = clean_plate_text(row.get("plate", ""))
            if row.get("file") and plate:
                texts[row["file"]] = plate
    return texts


def evaluate_image(entry: dict, root: Path, detector: PlateDetector, ocr: PlateOCR | None,
                   texts: dict[str, str]) -> ImageRow:
    gt = gt_boxes(entry)
    row = ImageRow(entry["file"], entry["split"], entry["vehicle"], len(gt), 0, "none", 0, 0,
                   truth=texts.get(entry["file"]))
    image = cv2.imdecode(np.fromfile(str(root / entry["file"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        row.error = "image could not be read"
        return row
    result = detector.detect(image)
    boxes = [d.bbox for d in result.detections]
    matches, false_positives = match_boxes(gt, boxes)
    row.predicted, row.method = len(boxes), result.method
    row.matched, row.false_positives = len(matches), len(false_positives)
    row.ious = [round(m[1], 3) for m in matches.values()]
    if ocr is not None:
        try:
            if result.best is not None:
                row.ocr_detected = OCRRow.from_result(read_ocr(ocr, result.best.crop))
            if gt:
                gt_crop = crop_with_padding(image, gt[0], detector.cfg.crop_padding)
                row.ocr_ground_truth = OCRRow.from_result(read_ocr(ocr, gt_crop))
        except Exception as exc:          # keep evaluating the other images
            row.error = f"OCR failed: {exc}"
    return row


def summarize(rows: list[ImageRow]) -> dict:
    """Aggregate numbers for the report (only from what was measured)."""
    def detection(subset: list[ImageRow]) -> dict:
        gt = sum(r.gt_plates for r in subset)
        tp = sum(r.matched for r in subset)
        pred = sum(r.predicted for r in subset)
        ious = [v for r in subset for v in r.ious]
        return {"images": len(subset), "gt_plates": gt, "detected": tp, "predicted_boxes": pred,
                "false_positives": sum(r.false_positives for r in subset),
                "recall": ratio(tp, gt), "precision": ratio(tp, pred),
                "mean_iou": float(np.mean(ious)) if ious else None,
                "images_all_found": sum(r.matched == r.gt_plates for r in subset)}

    def ocr_stats(subset: list[ImageRow], attr: str) -> dict:
        reads = [getattr(r, attr) for r in subset if getattr(r, attr) is not None]
        return {"reads": len(reads), "valid_plate": sum(o.plate is not None for o in reads),
                "accepted_without_review": sum(o.plate is not None and not o.needs_review for o in reads)}

    def accuracy(subset: list[ImageRow], attr: str) -> dict | None:
        annotated = [r for r in subset if r.truth]
        if not annotated:
            return None
        predictions = [getattr(r, attr).plate if getattr(r, attr) else None for r in annotated]
        return {"annotated_images": len(annotated),
                "plate_accuracy": ratio(sum(p == r.truth for p, r in zip(predictions, annotated)), len(annotated)),
                "char_accuracy": float(np.mean([char_accuracy(p, r.truth)
                                                for p, r in zip(predictions, annotated)]))}

    by_vehicle = {v: [r for r in rows if r.vehicle == v] for v in VEHICLES}
    by_method = defaultdict(list)
    for r in rows:
        by_method[r.method].append(r)
    methods = {m: {"images": len(rs), "detected": sum(r.matched for r in rs),
                   "false_positives": sum(r.false_positives for r in rs)} for m, rs in by_method.items()}
    gt_total = sum(r.gt_plates for r in rows)
    return {
        "detection": detection(rows),
        "detection_by_vehicle": {v: detection(rs) for v, rs in by_vehicle.items() if rs},
        "detection_by_method": methods,
        "yolo_only_recall": ratio(methods.get("yolo", {}).get("detected", 0), gt_total),
        "ocr_on_detected_box": ocr_stats(rows, "ocr_detected"),
        "ocr_on_ground_truth_box": ocr_stats(rows, "ocr_ground_truth"),
        "ocr_by_vehicle": {v: ocr_stats(rs, "ocr_ground_truth") for v, rs in by_vehicle.items() if rs},
        "ocr_accuracy_detected_box": accuracy(rows, "ocr_detected"),
        "ocr_accuracy_ground_truth_box": accuracy(rows, "ocr_ground_truth"),
        "errors": [f"{r.file}: {r.error}" for r in rows if r.error],
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def markdown_report(summary: dict, rows: list[ImageRow], meta: dict) -> str:
    d = summary["detection"]
    lines = [
        f"# Evaluation – {meta['tag']}",
        "",
        f"Generated by `python -m scripts.evaluate` on {meta['date']}.",
        "",
        f"* Dataset: [{meta['dataset']}]({DATASET_URL}) ({LICENSE}), splits: {', '.join(meta['splits'])}",
        f"* Weights: `{meta['model']}` (SHA-256 `{meta['model_sha256'][:12]}…`)",
        f"* Device: {meta['device']} · YOLO confidence {meta['confidence']} · input size {meta['image_size']} · "
        f"IoU threshold {IOU_THRESHOLD}",
        f"* Classical fallback: {'on' if meta['fallback'] else 'off'} · time: {meta['seconds']:.0f} s",
        "",
        "## Detection",
        "",
        "| Group | Images | Plates | Detected | Recall | Precision | Mean IoU | False positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def det_row(name: str, s: dict) -> str:
        iou_text = "—" if s["mean_iou"] is None else f"{s['mean_iou']:.2f}"
        return (f"| {name} | {s['images']} | {s['gt_plates']} | {s['detected']} | {_pct(s['recall'])} | "
                f"{_pct(s['precision'])} | {iou_text} | {s['false_positives']} |")

    lines.append(det_row("**All**", d))
    for vehicle, s in summary["detection_by_vehicle"].items():
        lines.append(det_row(vehicle, s))
    lines += ["", f"YOLO alone (without the classical fallback) detected "
                  f"{_pct(summary['yolo_only_recall'])} of the plates.", "",
              "| Method that produced the boxes | Images | Plates detected | False positives |",
              "|---|---:|---:|---:|"]
    for method, s in sorted(summary["detection_by_method"].items()):
        lines.append(f"| {method} | {s['images']} | {s['detected']} | {s['false_positives']} |")

    lines += ["", "## OCR", "",
              "The dataset has **no plate text annotation**, so without `plate_texts.csv` there is no "
              "ground truth and no accuracy figure. The table only counts what the OCR produced.", "",
              "| Crop read | Reads | Valid Brazilian plate | Accepted without manual review |",
              "|---|---:|---:|---:|"]
    for name, key in (("Box chosen by the detector", "ocr_on_detected_box"),
                      ("Ground-truth box", "ocr_on_ground_truth_box")):
        s = summary[key]
        lines.append(f"| {name} | {s['reads']} | {s['valid_plate']} ({_pct(ratio(s['valid_plate'], s['reads']))}) | "
                     f"{s['accepted_without_review']} ({_pct(ratio(s['accepted_without_review'], s['reads']))}) |")
    lines += ["", "Ground-truth box by vehicle type:", "", "| Vehicle | Reads | Valid plate | Accepted |",
              "|---|---:|---:|---:|"]
    for vehicle, s in summary["ocr_by_vehicle"].items():
        lines.append(f"| {vehicle} | {s['reads']} | {s['valid_plate']} | {s['accepted_without_review']} |")
    for name, key in (("detector box", "ocr_accuracy_detected_box"), ("ground-truth box", "ocr_accuracy_ground_truth_box")):
        acc = summary[key]
        if acc:
            lines += ["", f"Accuracy on the {acc['annotated_images']} human-annotated images ({name}): "
                          f"plate {_pct(acc['plate_accuracy'])}, character {_pct(acc['char_accuracy'])}."]

    missed = [r for r in rows if r.matched < r.gt_plates]
    lines += ["", "## Errors", "", f"Images with at least one missed plate: {len(missed)} of {len(rows)}.", ""]
    if missed:
        lines += ["| Image | Vehicle | Plates | Found | Method | Boxes predicted |", "|---|---|---:|---:|---|---:|"]
        lines += [f"| `{r.file}` | {r.vehicle} | {r.gt_plates} | {r.matched} | {r.method} | {r.predicted} |"
                  for r in missed]
    if summary["errors"]:
        lines += ["", "Processing errors:", ""] + [f"* {e}" for e in summary["errors"]]
    return "\n".join(lines) + "\n"


def export_crops(entries: list[dict], root: Path, padding: float) -> Path:
    """Write ground-truth crops and an empty ``plate_texts`` template for human annotation."""
    out = root / "crops"
    out.mkdir(exist_ok=True)
    template = root / "plate_texts_template.csv"
    with template.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["file", "plate", "crop"])
        for entry in entries:
            boxes = gt_boxes(entry)
            if not boxes:
                continue
            image = cv2.imdecode(np.fromfile(str(root / entry["file"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            crop = crop_with_padding(image, boxes[0], padding)
            name = Path(entry["file"]).stem + "_plate.png"
            scale = max(1, round(120 / max(1, crop.shape[0])))
            cv2.imencode(".png", cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))[1] \
                .tofile(str(out / name))
            writer.writerow([entry["file"], "", f"crops/{name}"])
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--split", choices=["all", "train", "valid", "test"], default="all")
    parser.add_argument("--model", type=Path, default=config.PLATE_MODEL_PATH)
    parser.add_argument("--tag", default="published", help="name of the report files")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONFIG.detection.confidence_threshold,
                        help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_CONFIG.detection.image_size,
                        help="YOLO input size")
    parser.add_argument("--no-fallback", action="store_true", help="disable the classical OpenCV detector")
    parser.add_argument("--no-ocr", action="store_true", help="evaluate detection only (faster)")
    parser.add_argument("--export-crops", action="store_true", help="write crops + template for annotation")
    args = parser.parse_args()

    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} not found. Run: python -m scripts.download_dataset")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    entries = [e for e in entries if e["split"] in splits]
    if args.export_crops:
        print(f"Template written: {export_crops(entries, args.dataset, config.DEFAULT_CONFIG.detection.crop_padding)}")
        return

    cfg = config.DetectionConfig(use_classical_fallback=not args.no_fallback, confidence_threshold=args.conf,
                                 image_size=args.imgsz)
    detector = PlateDetector(model_path=args.model, cfg=cfg)
    if not detector.yolo_available:
        raise SystemExit(detector.load_error)
    ocr = None if args.no_ocr else PlateOCR()
    if ocr is not None and not ocr.available:
        raise SystemExit(ocr.load_error)
    texts = load_texts(args.dataset)

    started = time.perf_counter()
    rows = []
    for index, entry in enumerate(entries, start=1):
        rows.append(evaluate_image(entry, args.dataset, detector, ocr, texts))
        if index % 50 == 0:
            print(f"  {index}/{len(entries)}")
    from modules.system_status import gpu_status, sha256_of
    gpu = gpu_status()
    meta = {"tag": args.tag, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "dataset": f"brazilian_lpr",
            "splits": splits, "model": args.model.name, "model_sha256": sha256_of(args.model),
            "device": gpu.device_name if gpu.usable else "CPU", "confidence": cfg.confidence_threshold,
            "fallback": cfg.use_classical_fallback, "image_size": cfg.image_size,
            "seconds": time.perf_counter() - started,
            "texts_annotated": len(texts)}
    summary = summarize(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_{args.split}"
    (REPORT_DIR / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "images": [asdict(r) for r in rows]}, indent=1),
        encoding="utf-8")
    (REPORT_DIR / f"{stem}.md").write_text(markdown_report(summary, rows, meta), encoding="utf-8")
    d = summary["detection"]
    print(f"Detection: {d['detected']}/{d['gt_plates']} plates (recall {_pct(d['recall'])}, "
          f"precision {_pct(d['precision'])}). Report: {REPORT_DIR / (stem + '.md')}")
    counts = Counter(r.method for r in rows)
    print("Methods:", dict(counts))


if __name__ == "__main__":
    main()
