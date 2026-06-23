# CLAUDE.md — M.A.L.L.I. Project Guide

## Project Overview

M.A.L.L.I. (Malaria Analysis via Low-cost Light Imaging) is a blood-smear malaria detection system combining:
- A Python/TensorFlow ML pipeline for training a MobileNetV3-small binary classifier
- A Flutter mobile app (Android only) for offline-friendly field capture and on-device inference

---

## Repository Layout

```
malli/
├── ml/                          ← ALL Python (canonical)
│   ├── src/
│   │   ├── data/
│   │   │   ├── loaders/
│   │   │   │   ├── nih_loader.py           MalariaDataset — NIH cell images
│   │   │   │   ├── synthetic_loader.py     SyntheticFieldReadyDataset
│   │   │   │   └── smear_roi_loader.py     SmearROILoader — patches from whole slides
│   │   │   ├── augmentation/
│   │   │   │   └── field_augment.py        CurriculumAugmentor + field transforms
│   │   │   └── preprocessing.py            Macenko stain normalisation
│   │   ├── models/
│   │   │   ├── factory.py                  build_mobilenetv3_small, compile_binary_model, F1Score
│   │   │   ├── inference.py                predict_image (threshold auto-resolved from weights dir)
│   │   │   ├── cell_counter.py             CellCounter (ROI sliding-window + classify)
│   │   │   ├── roi_detection/
│   │   │   │   ├── roi_grabber.py          Watershed/Otsu segmentation, ROI proposals, NMS
│   │   │   │   └── roi_grabber_yolo.py     YOLOv8 ROI variant (desktop research only)
│   │   │   ├── detection_pipeline.py       SAHI tiling inference
│   │   │   └── evaluation.py               calibrate_decision_threshold, evaluate_model
│   │   ├── export/
│   │   │   ├── pipeline.py                 Full export orchestrator (SavedModel → TFLite → assets)
│   │   │   ├── to_tflite.py                export_tflite_from_weights (INT8 quantization)
│   │   │   └── mobile_assets.py            MobileAssetManager, MobileAssetSyncPipeline
│   │   └── utils/
│   │       ├── visualization.py            LiveDashboardCallback, TensorBoard launcher
│   │       └── diagnose_tflite.py          TFLite evaluator on NIH / synthetic test sets
│   ├── tests/
│   │   ├── conftest.py                     Shared fixtures (fake images, NIH/smear fixture dirs)
│   │   ├── test_model_factory.py           MobileNetV3 builder + F1Score metric
│   │   ├── test_roi_pipeline.py            ROI segmentation, NMS, CellCounter init
│   │   ├── test_data_loaders.py            MalariaDataset, SmearROILoader, stratified split
│   │   └── test_export.py                  TFLite file creation, MobileAssetSyncPipeline checksum
│   ├── train.py                            7-stage curriculum training entry point
│   ├── evaluate.py                         Model evaluation script
│   └── requirements.txt                    Single pinned requirements file
│
├── mobile/                      ← ALL Flutter / Dart (Android only)
│   ├── lib/
│   │   ├── main.dart
│   │   ├── models/sample.dart              Sample dataclass (includes totalCells, infectedCells)
│   │   ├── screens/
│   │   │   ├── home_screen.dart
│   │   │   ├── capture_screen.dart         Wired to BloodSmearAnalyzer
│   │   │   └── camera_capture_screen.dart
│   │   ├── services/
│   │   │   ├── tflite_service.dart         MobileNetV3 INT8 TFLite inference
│   │   │   ├── blood_smear_analyzer.dart   On-device smear analysis (tile → classify → NMS)
│   │   │   ├── image_processor.dart        Image loading utilities, legacy analyzeBloodSmear
│   │   │   └── detection_pipeline.dart     BoundingBox, ImagePreProcessor, TilingInferenceEngine, NMS
│   │   └── database/
│   │       └── database_helper.dart        SQLite v2 (adds totalCells, infectedCells columns)
│   ├── assets/models/
│   │   └── malaria_detector.tflite         Synced by ml export pipeline; .gitkeep committed
│   ├── test/widget_test.dart
│   └── pubspec.yaml                        tflite_flutter ^0.10.4, sqflite, camera, image
│
├── datasets/                    ← All data (gitignored for large files)
│   ├── nih/                     NIH cell images (Parasitized / Uninfected)
│   ├── synthetic_field_ready/   Augmented synthetic cell crops
│   └── blood_smear/             Whole blood smear slides + labels.csv
│
├── experiments/                 ← Training artifacts; checkpoints, logs, decision_threshold.json
├── docs/                        ← Project documentation
├── CLAUDE.md
├── README.md
├── environment.yml
└── .gitignore
```

---

## Build & Run Commands

### Python / ML

```bash
# Activate environment
conda activate malli
# OR on Windows
.venv\Scripts\activate

# Install deps
pip install -r ml/requirements.txt

# Run full 7-stage curriculum training
python ml/train.py

# Override config
python ml/train.py --config path/to/override.json

# Smoke test — run only stages 1 & 2 for 2 epochs each
python ml/train.py --stages 1 2 --epochs 2 2 --export-disabled

# Evaluate saved weights
python ml/evaluate.py \
  --weights experiments/runs/latest/checkpoints/best.h5

# Run test suite
cd ml && pytest tests/ -v --tb=short
cd ml && pytest tests/ -v --tb=short --cov=src/
```

### Flutter / Mobile

```bash
cd mobile

# Install deps
flutter pub get

# Static analysis
flutter analyze

# Run widget tests
flutter test

# Run on device / emulator (Android only)
flutter run --debug
flutter run --release
```

### End-to-End Export & Sync

```bash
# After training: export INT8 TFLite and sync to mobile assets
python ml/train.py --export-only \
  --weights experiments/runs/latest/checkpoints/best.h5 \
  --sync-mobile mobile/

# Verify model was copied
ls mobile/assets/models/malaria_detector.tflite
```

---

## Training Curriculum

Seven stages across three phases:

| Phase | Stage | Dataset | Epochs | LR | Backbone |
|-------|-------|---------|--------|----|----------|
| A — Foundation | 1 | NIH cells | 5 | 1e-3 | Frozen |
| A — Foundation | 2 | NIH cells | 10 | 2e-4 | Unfrozen |
| B — Obscuring | 3 | NIH + curriculum augment | 8 | 5e-5 | Unfrozen |
| B — Obscuring | 4 | Synthetic field-ready | 8 | 2e-5 | Unfrozen |
| C — Smear ROI | 5 | Blood smear ROI patches | 10 | 1e-5 | Frozen |
| C — Smear ROI | 6 | Smear ROI + field augment | 8 | 5e-6 | Unfrozen |
| C — Smear ROI | 7 | Joint NIH + smear ROI | 5 | 1e-6 | Unfrozen |

---

## Architecture

### ML Data Flow

```
datasets/nih/  +  datasets/synthetic_field_ready/  +  datasets/blood_smear/
    │                       │                                │
MalariaDataset      SyntheticFieldReadyDataset        SmearROILoader
    │                       │                          (ROI extraction +
    └───────────────────────┴──────── TFRecord cache)  TFRecord cache)
                                              │
                                        model.fit()  (7-stage curriculum)
                                              │
                            experiments/.../best.weights.h5
                                              │
                              INT8 TFLite quantization (val split calibration)
                                              │
                            mobile/assets/models/malaria_detector.tflite
```

### Mobile Inference Flow

```
Camera → JPEG → BloodSmearAnalyzer.analyze()
                    │
          ImagePreProcessor (green-channel enhance)
                    │
          TilingInferenceEngine (128px overlapping ROI tiles)
                    │
          TFLiteService.classifyCell() per tile
          (INT8 dequantization via scale/zeroPoint from model)
                    │
          PostInferenceFilter.applyNMS()
                    │
          parasitemiaPercent = infectedCells / totalCells × 100
                    │
          DatabaseHelper (SQLite v2) → home_screen.dart
```

---

## Key Implementation Notes

- **NumPy pinned to 1.x**: `numpy==1.24.3` — must stay on 1.x for TF 2.13.1 ABI compatibility
- **INT8 calibration from val split**: `export_tflite_from_weights` uses `calibration_dataset` (not training data)
- **Decision threshold**: Loaded from `decision_threshold.json` alongside weights; defaults to 0.3 (recall-first). `inference.py` resolves it automatically relative to the weights path when not specified.
- **TFRecord cache versioning**: `MalariaDataset._CACHE_SCHEMA_VERSION = 2` — changing this number forces full cache rebuild
- **Stain normalisation**: Macenko SVD-based, implemented in `ml/src/data/preprocessing.py`; enabled by default in both NIH loader and SmearROILoader
- **YOLO ROI**: `roi_grabber_yolo.py` is research-only (Python desktop); excluded from mobile assets
- **Android only**: TFLite INT8 export only; no CoreML, no iOS target

---

## Dataset Paths

| Dataset | Local path | Notes |
|---------|-----------|-------|
| NIH cell images | `datasets/nih/cell_images/` | Parasitized/ and Uninfected/ subdirs |
| Synthetic field-ready | `datasets/synthetic_field_ready/` | With `labels.csv` |
| Blood smear slides | `datasets/blood_smear/` | With `labels.csv` (filename, label) |

All dataset directories are gitignored. Only the `datasets/` directory structure is committed.
