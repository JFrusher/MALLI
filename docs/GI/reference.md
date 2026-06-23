M.A.L.L.I. Reference
=====================

Purpose
-------
This file explains the end-to-end training process, key files, metrics, logs, and what to watch for during an experiment. Use it as a quick reference when running `python train.py` or reviewing results.

Quick Index
-----------
- Overview of training flow
- What runs on `python train.py`
- Data pipelines (NIH, synthetic, TFRecord cache)
- Per-epoch lifecycle and what each metric means
- Where logs live and what to inspect
- Metrics glossary and plain-language explainer you can use in conversations
- Troubleshooting notes (OOM, Unknown steps, export failures)

Overview
--------
`train.py` implements a staged curriculum: it runs several training "stages" sequentially. Each stage may use either the real NIH dataset or the synthetic "field-ready" dataset. We build datasets once at startup and reuse them across stages.

Control flow (high level)
-------------------------
1. `main()` loads configuration and initializes logging + TensorBoard directories.
2. `build_dataset_registry()` constructs two datasets:
   - NIH dataset: `data.data_loader.MalariaDataset` (prefers TFRecord cache)
   - Synthetic dataset: `data.synthetic_data_loader.SyntheticFieldReadyDataset`
3. For each stage in `DEFAULT_CONFIG['stages']`:
   - Build and compile a fresh MobileNetV3 model
   - Optionally load previous stage weights
   - Train with `model.fit(train_ds, validation_data=val_ds, callbacks=...)`
   - Save best weights for that stage
4. After all stages: evaluate holdouts, calibrate threshold, save final model, export TFLite.

Data pipelines
--------------
- NIH pipeline (`data/data_loader.py`)
  - Preferred path: compressed TFRecord shards under `datasets/blood_smear/processed/tfrecord/`.
  - If the cache is missing, the code will materialize it from `dataverse_files.zip` or extracted folders.
  - TFRecords contain raw image bytes, integer label, and a sample weight.
  - The TFRecord dataset advertises exact cardinality so Keras shows `steps_per_epoch` (no more `/Unknown`).

- Synthetic pipeline (`data/synthetic_data_loader.py`)
  - Reads `synthetic_field_ready/labels.csv` and loads images as needed.
  - Supports soft labels and per-sample weights during training.

Per-epoch lifecycle
-------------------
- `model.fit()` consumes training batches in a loop; each batch is:
  - decoded from bytes (if using TFRecord) or read from file
  - converted to float32 and resized to `image_size`
  - stain-normalized (placeholder) and augmented if training
  - batched and prefetched to speed up the GPU/CPU
- After each epoch, the validation dataset is evaluated to compute `val_*` metrics.
- Callbacks:
  - `ModelCheckpoint`: saves stage-best weights when the monitored metric (`val_auc`) improves
  - `EarlyStopping`: stops the stage early when monitored metric stops improving
  - `ReduceLROnPlateau`: reduces learning rate to continue progress
  - `TensorBoard` and `CSVLogger`: produce logs for visualization and CSV results

Metrics glossary
----------------
- `loss`: binary cross-entropy; lower is better.
- `accuracy`: (TP + TN) / total; can be misleading with class imbalance.
- `precision`: TP / (TP + FP) — how many predicted positives are correct.
- `recall` (sensitivity): TP / (TP + FN) — how many true positives are found.
- `f1_score`: harmonic mean of precision and recall.
- `auc`: area under ROC curve; measures ranking/separability.
- `val_*`: same metrics measured on the holdout set.

Where logs and artifacts live
----------------------------
- Checkpoints: `models/` (e.g., `stage_1_nih_warmup.weights.h5`)
- Final model: `models/last_mobilenetv3_small.keras`
- TFLite export: `models/mobilenetv3_small_int8.tflite`
- TensorBoard logs: `logs/tensorboard/<timestamp>/<stage_name>/` and root `logs/tensorboard` for all runs
- CSV per-stage metrics: `logs/metrics_<stage_name>.csv`
- TFRecord cache: `datasets/blood_smear/processed/tfrecord/`

Simple plain-language explainer (copyable)
-----------------------------------------
"The training script runs a short curriculum: first it warms up on the real NIH single-cell images to learn basic features, then it refines the network on a larger set of synthetic, field-like images so the model generalizes to low-cost optics. Each stage trains for a few epochs with validation checks; the best weights are saved per stage. At the end we calibrate a decision threshold and export an INT8 TFLite model for mobile deployment."

Troubleshooting notes
---------------------
- OOM on CPU/GPU: reduce `batch_size` in `train.py` or use a smaller input resolution.
- `/Unknown` in TensorFlow progress: rebuild TFRecord cache so `cache_manifest.json` exists or set `rebuild_cache=True`.
- TFLite quantization fails: ensure representative dataset yields `float32` images (generator handles both weighted and unweighted batches).

---
Generated on: 2026-05-14
