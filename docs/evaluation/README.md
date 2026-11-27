# Evaluation on real Brazilian plates

All numbers in this folder were produced by the scripts in `scripts/` on real
images; nothing was estimated or typed by hand. Reproduce with:

```bash
set ROBOFLOW_API_KEY=...                 # PowerShell: $env:ROBOFLOW_API_KEY="..."
python -m scripts.download_dataset       # 370 images -> datasets/brazilian_lpr (outside git)
python -m scripts.evaluate               # -> docs/evaluation/published_all.{md,json}
```

## Dataset

[brazilian_lpr](https://universe.roboflow.com/lpr-rn1pv/brazilian_lpr) (Roboflow
Universe, workspace `lpr-rn1pv`, **CC BY 4.0**). The project has no published
version, so it cannot be exported the usual way; `scripts/download_dataset.py`
rebuilds it from the image API, keeping the authors' split.

| Split | Images | Motorcycle | Car | Unknown |
|---|---:|---:|---:|---:|
| train | 294 | 272 | 10 | 12 |
| valid | 64 | 60 | 3 | 1 |
| test | 12 | 7 | 3 | 2 |

What this means for the results:

* **It is mostly motorcycles** (92 %), photographed by a traffic camera. A
  gate camera sees cars, close and centred, so these numbers are a *hard*
  scenario for the system, not its expected performance at a gate.
* Most images were resized by Roboflow to 416 × 416; the median plate is
  **35 px wide**. Many motorcycle plates are unreadable even for a person.
* The annotation has **boxes only, no plate text**. Plate-level and
  character-level OCR accuracy therefore **cannot be measured** from this
  dataset. `scripts/evaluate.py` computes them automatically if a person fills
  `datasets/brazilian_lpr/plate_texts.csv` (`file;plate`);
  `python -m scripts.evaluate --export-crops` writes the plate crops and an
  empty template to make that annotation quick.

## Results with the published YOLO11 weights

Full report with every missed image: [`published_all.md`](published_all.md)
(per-image data in `published_all.json`). RTX 5050, confidence 0.25, input
640, IoU ≥ 0.5.

| Group | Plates | Detected | Recall | Precision | Mean IoU |
|---|---:|---:|---:|---:|---:|
| All | 418 | 248 | 59.3 % | 94.3 % | 0.83 |
| Car | 16 | 14 | 87.5 % | 100 % | 0.86 |
| Motorcycle | 387 | 222 | 57.4 % | 93.7 % | 0.83 |

* The classical OpenCV fallback ran on the 123 images where YOLO found
  nothing and **did not recover any plate** (no false positives either): its
  character-count verification rejects plates this small.
* OCR (no ground truth, so only what it produced): on the box chosen by the
  detector, 94 of 247 reads (38 %) were a valid Brazilian plate and 60 (24 %)
  would be accepted without manual review. A *valid* plate is not necessarily
  the *correct* one; e.g. the bus plate `BAH-5201` (demo image 04) was read
  `OAH5201`, but with 19 % confidence, so it goes to manual review instead of
  being accepted.

### Threshold experiment (train split, detection only)

Chosen on the train split so that valid/test stay untouched.

| YOLO confidence | Input size | Recall | Precision |
|---:|---:|---:|---:|
| 0.25 (default) | 640 | 59.7 % | 93.5 % |
| 0.25 | 960 | 60.3 % | 91.5 % |
| 0.25 | 1280 | 62.4 % | 87.2 % |
| 0.10 | 640 | 67.1 % | 90.5 % |
| 0.10 | 960 | 66.5 % | 86.6 % |
| 0.10 | 1280 | 70.3 % | 79.9 % |

A lower threshold finds more of the tiny motorcycle plates, but those are the
plates the OCR cannot read, and every extra false box is read by the OCR. The
default (0.25 / 640) was kept for the gate scenario; the threshold can be
changed on the Settings screen. The real fix for small plates is to train the
detector on this kind of image (see `scripts/train_detector.py`).

## Fine-tuning (item 11)

`python -m scripts.train_detector --install` fine-tunes the published YOLO11s
on the train split (50 epochs, imgsz 640, `fliplr=0`, 4.9 min on the RTX 5050).
Full table in [`finetune.md`](finetune.md); YOLO's own validation (`model.val`):

| Split | Metric | Published | Fine-tuned |
|---|---|---:|---:|
| valid (64 img) | mAP@0.5 | 0.750 | 0.985 |
| valid | mAP@0.5:0.95 | 0.505 | 0.816 |
| valid | Recall | 0.590 | 0.984 |
| test (12 img) | mAP@0.5 | 0.903 | 0.995 |
| test | mAP@0.5:0.95 | 0.572 | 0.810 |

**About the published model's own metrics:** its [model card](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
warns that the upstream dataset has train/test contamination (the same source
images in both splits), that the published metrics are therefore inflated, and
it reports them only for the YOLO11x variant. None of those figures is used
here; the "Published" column was measured again on Brazilian images.

The whole system (`scripts/evaluate.py`, detector + OCR), same splits:

| Split | Model | Plates detected | Precision | OCR valid plate | Accepted without review |
|---|---|---:|---:|---:|---:|
| valid | published | 35 / 66 | 97.2 % | 15 / 36 | 11 |
| valid | fine-tuned | 64 / 66 | 98.5 % | 16 / 62 | 11 |
| test | published | 10 / 12 | 100 % | 3 / 10 | 1 |
| test | fine-tuned | 12 / 12 | 100 % | 3 / 12 | 2 |

What this shows:

* Detection of motorcycle plates in these scenes goes from about half to almost
  all. The bus in demo image 04 is now read correctly (`BAH5201`, 72 %)
  because the box is tighter.
* **The OCR does not follow.** The plates recovered by the fine-tuned detector
  are the tiny ones (≈ 30 px); the OCR cannot read them, so they still end in
  manual review. At this resolution the bottleneck is the OCR, not the detector.
* **Regression: false positives outside these scenes.** The dataset has no
  background images (every image contains a plate), and almost all plates are
  small squarish motorcycle plates. The fine-tuned model fires (confidence
  0.54) on a 25 × 30 px region of Ultralytics' `zidane.jpg`, a photo without
  any plate; the published model does not.
* The splits share the same fixed cameras (checked: no vehicle appears in two
  splits, but the scenes are the same), and the best epoch is chosen on valid,
  so the fine-tuned numbers are optimistic for other places.

**Decision:** the app keeps the **published** weights by default. The
fine-tuned file is installed as `models/plate_detector_finetuned.pt` and is
used only with `SMART_GATE_MODEL=finetuned`. Next steps if the group wants to
adopt it: add background images (no plates) and car photos from the gate
camera to the training set, then repeat `train_detector` + `evaluate`.

## Problems found and what was done

| Problem | Action |
|---|---|
| RTX 50xx (Blackwell, `sm_120`) is not supported by the PyTorch cu124 build in the setup | Documented in `requirements.txt` (use the cu128 build); the Settings screen detects the mismatch |
| Dataset has no exportable version | Download rebuilt from the image API |
| Motorcycle plates far from the camera are missed | Fine-tuning (item 11) detects them; the OCR still cannot read them, so they go to manual review, never to "authorized" |
| Fine-tuned detector fires on a photo without plates | Kept the published model as default; regression test pinned per model |
| Wrong but *valid-looking* reads | Confirmed that low confidence sends them to manual review (regression test in `tests/test_demo_images.py`) |
| No text annotation → no OCR accuracy | Not invented: support for a human-made `plate_texts.csv` instead |
