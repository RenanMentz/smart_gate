"""Tests for the dataset download (annotation conversion) and the evaluation metrics.

No network access and no model: the Roboflow payloads are small hand-written
examples with the same shape as the API responses.
"""

import pytest

from scripts import download_dataset as dd
from scripts import evaluate as ev


# ------------------------------------------------------------ annotations
@pytest.mark.parametrize(("label", "expected"), [
    ("PLATE", True), ("plate", True), ("placa moto mercosul", True), ("placa carro", True), ("CAR", False),
])
def test_is_plate_label(label, expected):
    assert dd.is_plate_label(label) is expected


@pytest.mark.parametrize(("labels", "vehicle"), [
    (["placa moto"], "motorcycle"), (["placa carro mercosul"], "car"), (["PLATE", "CAR"], "car"),
    (["PLATE"], "unknown"),
])
def test_vehicle_type(labels, vehicle):
    assert dd.vehicle_type(labels) == vehicle


def test_parse_box_rectangle_and_polygon():
    box = dd.parse_box({"label": "PLATE", "x": "100.0", "y": "50.0", "width": "40.0", "height": "20.0"})
    assert box.xyxy == (80, 40, 120, 60)
    poly = dd.parse_box({"label": "plate", "type": "polygon", "x": "0", "y": "0", "width": "0", "height": "0",
                         "points": [[10, 20], [50, 22], [48, 40], [12, 38]]})
    assert poly.xyxy == (10, 20, 50, 40)


def test_yolo_lines_are_normalized_and_clipped():
    inside = dd.PlateBox(100, 50, 40, 20, "PLATE")
    outside = dd.PlateBox(195, 50, 20, 20, "PLATE")       # half outside the 200 px wide image
    lines = dd.yolo_lines([inside, outside], 200, 100)
    assert lines[0] == "0 0.500000 0.500000 0.200000 0.200000"
    cls, cx, cy, w, h = lines[1].split()
    assert cls == "0" and float(cx) + float(w) / 2 == pytest.approx(1.0)
    assert dd.yolo_lines([dd.PlateBox(500, 500, 10, 10, "PLATE")], 200, 100) == []   # fully outside


def test_safe_name_is_unique_per_image():
    assert dd.safe_name("abcdefgh123", "foto 1.png") == "foto_1_abcdefgh.jpg"
    assert dd.safe_name("zzzzzzzz999", "foto 1.png") != dd.safe_name("abcdefgh123", "foto 1.png")


def test_data_yaml_has_single_plate_class(tmp_path):
    text = dd.data_yaml(tmp_path)
    assert "val: valid/images" in text and "0: plate" in text and "CC BY 4.0" in text


# ----------------------------------------------------------------- metrics
def test_match_boxes_one_to_one():
    gt = [(0, 0, 100, 50), (200, 0, 300, 50)]
    predicted = [(5, 0, 105, 50), (2, 0, 100, 48), (500, 500, 520, 510)]   # 2nd overlaps an already matched plate
    matches, false_positives = ev.match_boxes(gt, predicted)
    assert set(matches) == {0} and matches[0][0] == 0 and matches[0][1] > 0.9
    assert false_positives == [1, 2]


def test_match_boxes_requires_iou_threshold():
    matches, false_positives = ev.match_boxes([(0, 0, 100, 100)], [(60, 0, 160, 100)])   # IoU 0.25
    assert matches == {} and false_positives == [0]


@pytest.mark.parametrize(("pred", "truth", "acc"), [
    ("ABC1D23", "ABC1D23", 1.0), ("ABC1B23", "ABC1D23", 6 / 7), ("ABC123", "ABC1D23", 6 / 7),
    (None, "ABC1D23", 0.0), ("XXXXXXXXXXXXXXX", "ABC1D23", 0.0),
])
def test_char_accuracy(pred, truth, acc):
    assert ev.char_accuracy(pred, truth) == pytest.approx(acc)


def test_levenshtein():
    assert ev.levenshtein("kitten", "sitting") == 3
    assert ev.levenshtein("", "abc") == 3


def make_row(**kwargs):
    base = dict(file="a.jpg", split="test", vehicle="car", gt_plates=1, predicted=1, method="yolo", matched=1,
                false_positives=0, ious=[0.8])
    base.update(kwargs)
    return ev.ImageRow(**base)


def test_summarize_counts_only_what_was_measured():
    rows = [
        make_row(ocr_ground_truth=ev.OCRRow("ABC1D23", "ABC1D23", 0.9, False), truth="ABC1D23"),
        make_row(file="b.jpg", vehicle="motorcycle", predicted=2, matched=0, false_positives=2, ious=[],
                 method="classical", ocr_ground_truth=ev.OCRRow(None, "AB", 0.2, True)),
        make_row(file="c.jpg", vehicle="motorcycle", predicted=0, matched=0, ious=[], method="none"),
    ]
    s = ev.summarize(rows)
    d = s["detection"]
    assert (d["gt_plates"], d["detected"], d["predicted_boxes"]) == (3, 1, 3)
    assert d["recall"] == pytest.approx(1 / 3) and d["precision"] == pytest.approx(1 / 3)
    assert s["yolo_only_recall"] == pytest.approx(1 / 3)
    assert s["detection_by_vehicle"]["motorcycle"]["detected"] == 0
    assert s["ocr_on_ground_truth_box"] == {"reads": 2, "valid_plate": 1, "accepted_without_review": 1}
    acc = s["ocr_accuracy_ground_truth_box"]
    assert acc["annotated_images"] == 1 and acc["plate_accuracy"] == 1.0
    assert s["ocr_accuracy_detected_box"]["plate_accuracy"] == 0.0     # no detector read for a.jpg


def test_without_text_annotation_there_is_no_accuracy():
    s = ev.summarize([make_row(ocr_ground_truth=ev.OCRRow("ABC1D23", "ABC1D23", 0.9, False))])
    assert s["ocr_accuracy_ground_truth_box"] is None
    meta = {"tag": "t", "date": "d", "dataset": "x", "splits": ["test"], "model": "m.pt", "model_sha256": "0" * 64,
            "device": "CPU", "confidence": 0.25, "image_size": 640, "fallback": True, "seconds": 1}
    report = ev.markdown_report(s, [make_row()], meta)
    assert "no plate text annotation" in report and "Accuracy on the" not in report


def test_load_texts_reads_the_human_csv(tmp_path):
    (tmp_path / ev.TEXTS_FILE).write_text("file;plate\na.jpg;abc-1d23\nb.jpg;\n", encoding="utf-8-sig")
    assert ev.load_texts(tmp_path) == {"a.jpg": "ABC1D23"}
    assert ev.load_texts(tmp_path / "missing") == {}
