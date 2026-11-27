# Demonstration images

## Credit and license

All images in this folder come from the public dataset **brazilian_lpr**,
published on Roboflow Universe by the workspace `lpr-rn1pv`:
<https://universe.roboflow.com/lpr-rn1pv/brazilian_lpr>

License: **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).
The images were renamed and are otherwise unmodified. They are used here only
to demonstrate the system; the plates belong to the vehicles photographed by
the dataset authors.

| File | Original file in the dataset (split) |
|---|---|
| `01_moto_mercosul.jpg` | `tt1.jpg` (test) |
| `02_moto_antiga.jpg` | `20210114_1605229700750.jpg` (train) |
| `03_moto_antiga.jpg` | `20210114_1606435912640.jpg` (train) |
| `04_onibus_camera_transito.jpg` | `6_220624105205_0000000_B.jpg` (test) |
| `05_moto_distante.jpg` | `20201229_132340467267_png.jpg` (test) |

## What the system does with each image

Real output of `RecognitionPipeline` with the published YOLO11 weights, the
default settings and an empty vehicle registry (so readable plates are
*denied*; register the plate on the Vehicles screen to see the gate open).

| File | Detection | OCR | Confidence | Decision | Why it is a useful demo |
|---|---|---|---:|---|---|
| `01_moto_mercosul.jpg` | YOLO | `ASS2193` | 97% | Denied (not registered) | Close-up plate, clean read |
| `02_moto_antiga.jpg` | YOLO | `NZM8373` | 98% | Denied (not registered) | Two-row motorcycle plate |
| `03_moto_antiga.jpg` | YOLO | `JSY8838` | 100% | Denied (not registered) | Two-row plate with city strip ignored |
| `04_onibus_camera_transito.jpg` | YOLO | `OAH5201` | 19% | **Manual review** | Wrong read (the plate is `BAH-5201`) caught by the confidence threshold |
| `05_moto_distante.jpg` | none | — | — | **Manual review** | Distant motorcycle: plate too small to be detected |

With the fine-tuned detector (`SMART_GATE_MODEL=finetuned`, see
`docs/evaluation/README.md`) images 01–03 give the same plates, image 04 is
read correctly as `BAH5201` (72 %) and image 05 is detected but unreadable
(manual review).

**With the current default OCR (fast-plate-ocr)** the table above changes:
image 04 is read correctly as `BAH5201`, and image 01 is read `ASS2I93`
(5th character wrong). The table above was recorded with EasyOCR, the previous
default.

Values recorded on 2026-11-26 (RTX 5050, PyTorch 2.7.1 + CUDA 12.8). Other
hardware may produce slightly different confidences.
