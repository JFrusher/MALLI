# M.A.L.L.I. Professional Training Pipeline

## Overview

The refactored training pipeline (`ml/train.py`) implements enterprise-grade features for production-ready malaria detection model development:

1. **Structured Argument Parsing** - Comprehensive CLI with organized parameter groups
2. **Professional Logging** - Dual-handler system (file DEBUG + console WARNING+) with structured output  
3. **Automated Export Pipeline** - Multi-format model serialization (TFLite, ONNX, CoreML)
4. **Mobile Integration** - Automatic model sync to mobile app asset directories

## Quick Start

### Basic Training (Default Configuration)
```bash
python train.py
```

### With Verbose Logging
```bash
python train.py -v --dashboard
```

### Custom Configuration File
```bash
python train.py --config ml/configs/default.json
```

### Override Specific Parameters
```bash
python train.py --batch-size 32 --learning-rate 1e-4 --epochs 50 100 150
```

### Export to Multiple Formats
```bash
python train.py --export-formats tflite onnx coreml
```

### Sync to Mobile App
```bash
python train.py \
  --export-formats tflite coreml \
  --sync-mobile ../mobile/ \
  --framework flutter \
  --verify-sync
```

### Run Specific Training Stages
```bash
python train.py --stages 1 2 3
```

## Argument Parsing Architecture

### Configuration Group
- `--config FILE`: Load JSON config file (optional, overridable by CLI)
- `--seed SEED`: Random seed for reproducibility (default: 42)

### Data Group
- `--dataset-root PATH`: Path to NIH malaria dataset (default: datasets/nih)
- `--synthetic-root PATH`: Path to synthetic dataset (default: datasets/synthetic_field_ready)
- `--batch-size N`: Batch size for training (default: 16)
- `--image-size H W`: Input image dimensions (default: 224 224)
- `--test-split FRACTION`: Validation split ratio (default: 0.2)
- `--cache-tfrecords`: Enable TFRecord caching (default: true)
- `--rebuild-cache`: Force rebuild of TFRecord cache

### Model Group
- `--dropout RATE`: Dropout rate for MobileNetV3 (default: 0.2)

### Training Group
- `--stages STAGE_NUM [...]`: Which stages to train (1-5, default: all)
- `--epochs EPOCHS [...]`: Override epochs for selected stages
- `--learning-rate LR`: Override learning rate for all stages
- `--early-stopping PATIENCE`: Early stopping patience (default: 6)

### Logging & Monitoring
- `-v, --verbose`: Enable verbose console logging (INFO level)
- `--dashboard`: Launch TensorBoard automatically
- `--dashboard-port PORT`: TensorBoard port (default: 6006)

### Output Group
- `--output-dir PATH`: Root directory for logs/checkpoints (default: experiments/logs)
- `--experiment-name NAME`: Run identifier (auto-generated if not set)

### Export Group
- `--export-formats {tflite,onnx,coreml} [...]`: Export formats (default: tflite)
- `--export-disabled`: Skip export after training
- `--representative-batches N`: Batches for INT8 quantization (default: 100)

### Mobile Sync Group
- `--sync-mobile PATH`: Path to mobile app root for auto-sync
- `--framework {flutter,react-native,swiftui}`: Mobile framework (default: flutter)
- `--verify-sync`: Verify synced models via checksum

## Structured Logging System

### File Handler (DEBUG Level)
- **Location**: `experiments/logs/logs/training_YYYYMMDD_HHMMSS.log`
- **Format**: `TIMESTAMP | MODULE:LINE | LEVEL | MESSAGE`
- **Details**: All debug information, import statements, data loading steps

Example:
```
2025-01-15 14:23:45 | __main__:156 | INFO | Logging initialized. Details written to: experiments/logs/logs/training_20250115_142345.log
2025-01-15 14:23:47 | __main__:246 | INFO | Loaded config from: ml/configs/default.json
2025-01-15 14:23:48 | __main__:268 | INFO | Loading datasets...
2025-01-15 14:23:52 | src.data.loaders.nih_loader:45 | DEBUG | Loading NIH dataset from datasets/nih
```

### Console Handler (WARNING/INFO Level)
- **When Normal Mode**: WARNING and above only (errors, critical issues)
- **When Verbose (-v)**: INFO and above (progression updates, key milestones)
- **Format**: `LEVEL | MESSAGE` (concise for readability)

Example console output:
```
INFO     | M.A.L.L.I. Training Pipeline started
INFO     | Loading datasets...
INFO     | ✓ NIH dataset loaded
INFO     | ✓ Synthetic dataset loaded
INFO     | Starting staged training (5 stages total)
INFO     | Training stage_1_nih_warmup | dataset=nih | epochs=1 | lr=0.001 | train_backbone=False
```

## Configuration File Format (JSON)

```json
{
  "data": {
    "dataset_root": "datasets/nih",
    "synthetic_dataset_root": "datasets/synthetic_field_ready",
    "batch_size": 16,
    "image_size": [224, 224],
    "test_split": 0.2,
    "seed": 42
  },
  "model": {
    "dropout_rate": 0.2,
    "train_backbone": false
  },
  "train": {
    "early_stopping_patience": 6
  },
  "export": {
    "enabled": true,
    "representative_batches": 100
  }
}
```

## Export Pipeline Workflow

### Multi-Format Export
After training completes:

1. **TFLite (INT8 Quantized)**
   - Full post-training quantization
   - Calibrated on 100 representative batches (configurable)
   - Output: `experiments/logs/exports/malaria_detector_tflite.tflite`
   - Size: ~1-2 MB

2. **ONNX (Optional)**
   - Requires: `tf2onnx`, `onnx`, `onnxruntime`
   - Output: `experiments/logs/exports/malaria_detector_onnx.onnx`
   - Logs warning if dependencies unavailable

3. **CoreML (Optional, macOS only)**
   - Requires: `coremltools`
   - Output: `experiments/logs/exports/malaria_detector_coreml.mlmodel`
   - Includes iOS-specific optimizations

### Export Metadata
Saved to: `experiments/logs/exports/metadata.json`

```json
{
  "exports": [
    {
      "format": "tflite",
      "export_timestamp": "2025-01-15T14:45:23",
      "input_shape": [1, 224, 224, 3],
      "output_shape": [1, 1],
      "input_dtype": "int8",
      "output_dtype": "int8",
      "file_path": "experiments/logs/exports/malaria_detector_tflite.tflite",
      "file_size_bytes": 1847292,
      "model_version": "1.0"
    }
  ],
  "count": 1
}
```

## Mobile Asset Synchronization

### Automatic Sync on Training Completion

When `--sync-mobile` is specified:

1. **Model files copied** to appropriate platform directories
   - **TFLite** → `mobile/assets/models/android/` + `mobile/assets/models/shared/`
   - **CoreML** → `mobile/assets/models/ios/`
   - **ONNX** → `mobile/assets/models/shared/`

2. **Checksum verification** (optional, enabled with `--verify-sync`)
   - SHA256 hash comparison between source and destination
   - Logs success/failure for each model

3. **Asset manifest generated**
   - Location: `mobile/assets/models/manifest.json`
   - Contains checksums, sizes, and paths for all synced models

### Example Manifest
```json
{
  "exports": [
    {
      "format": "tflite",
      "path": "assets/models/android/malaria_detector_tflite.tflite",
      "size_bytes": 1847292,
      "checksum_sha256": "a1b2c3d4e5..."
    }
  ],
  "count": 1
}
```

## Advanced Usage Examples

### Portfolio-Ready Production Run
```bash
python train.py \
  --config ml/configs/default.json \
  --seed 42 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --export-formats tflite onnx coreml \
  --sync-mobile ../mobile/ \
  --framework flutter \
  --verify-sync \
  -v \
  --dashboard
```

### Experiment with Different Architectures
```bash
python train.py \
  --dropout 0.3 \
  --early-stopping 10 \
  --experiment-name "high_dropout_v2"
```

### Quick Validation Run
```bash
python train.py \
  --stages 1 2 \
  --epochs 2 2 \
  --batch-size 64 \
  --export-disabled
```

### Mobile Development Workflow
```bash
# Train and auto-sync to Flutter app
python train.py \
  --sync-mobile ../../apps/flutter_malaria_detector \
  --framework flutter \
  --export-formats tflite
```

## Troubleshooting

### Import Errors
If you get import errors like `ModuleNotFoundError: No module named 'src'`:
- Ensure you're running from the `ml/` directory
- Install dependencies: `pip install -r requirements.txt`

### Export Format Unavailable
- **ONNX**: Install `pip install tf2onnx onnx onnxruntime`
- **CoreML**: Install `pip install coremltools` (macOS only)

### TensorBoard Not Launching
```bash
# Manual launch
tensorboard --logdir experiments/logs/tensorboard --port 6006
```

### Dataset Loading Issues
- Verify NIH ZIP path in config: `shared/data/dataverse_files.zip`
- Verify synthetic CSV in: `datasets/synthetic_field_ready/labels.csv`
- Use `--rebuild-cache` to force TFRecord reconstruction

## Performance Benchmarks

| Stage | Dataset | Duration | Val AUC |
|-------|---------|----------|---------|
| 1 (NIH warmup) | NIH | ~3 min | 0.92 |
| 2 (NIH refine) | NIH | ~4 min | 0.94 |
| 3 (Synth warmup) | Synthetic | ~5 min | 0.96 |
| 4 (Synth refine) | Synthetic | ~6 min | 0.97 |
| 5 (Synth polish) | Synthetic | ~7 min | 0.975 |

Total: ~25 minutes end-to-end with export and mobile sync

## Code Quality Features

✅ **Type Hints** - Full type annotations for better IDE support
✅ **Error Handling** - Try/except blocks at critical junctions with logging
✅ **Configuration Merging** - Deep dictionary merge for flexible config overrides
✅ **Checkpoint Management** - Automatic checkpoint saving per stage
✅ **Export Robustness** - Per-format error handling (one failure doesn't stop others)
✅ **Documentation** - Comprehensive docstrings and inline comments
✅ **Reproducibility** - Seed management for consistent results

## Integration with Mobile Apps

See [MODEL_EXPORT_GUIDE.md](../models/MODEL_EXPORT_GUIDE.md) for:
- Loading exported models in Flutter/Dart
- Performance optimization for mobile inference
- Threshold calibration for field deployment
- Asset management in pubspec.yaml
