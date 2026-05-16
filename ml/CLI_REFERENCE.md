# M.A.L.L.I. Training Pipeline - CLI Quick Reference

## View Full Help
```bash
python train.py --help
```

## Most Common Commands

### 1. Default Training (recommended start)
```bash
python train.py
```
- Uses default config
- Standard logging to console
- No export (to speed up iteration)

### 2. Production Run with All Features
```bash
python train.py \
  --config configs/default.json \
  --export-formats tflite \
  --sync-mobile ../mobile/ \
  --framework flutter \
  -v \
  --dashboard
```

### 3. Development with Verbose Output
```bash
python train.py -v --dashboard
```
- Verbose console logging (INFO level)
- Dashboard launched automatically
- Useful for debugging data loading

### 4. Export After Training
```bash
python train.py --export-formats tflite onnx coreml
```
- Exports to all three formats
- Syncs to mobile if configured
- Metadata saved for tracking

### 5. Quick Validation (test only)
```bash
python train.py --stages 1 2 --epochs 2 2
```
- Only runs first 2 training stages
- 2 epochs each (instead of full schedule)
- Fast way to verify pipeline works

### 6. Custom Batch Size & Learning Rate
```bash
python train.py --batch-size 32 --learning-rate 1e-4
```

### 7. Mobile Development Workflow
```bash
python train.py \
  --sync-mobile ../../projects/malaria_app \
  --framework flutter \
  --export-formats tflite
```

---

## Argument Reference by Category

### Configuration
```
--config FILE              Load JSON config (optional)
--seed SEED               Random seed (default: 42)
```

### Data
```
--dataset-root PATH       NIH dataset path (default: datasets/nih)
--synthetic-root PATH     Synthetic dataset path (default: datasets/synthetic_field_ready)
--batch-size N            Batch size (default: 16)
--image-size H W          Image dimensions (default: 224 224)
--test-split FRAC         Validation split (default: 0.2)
--rebuild-cache           Force rebuild of TFRecord cache
```

### Training
```
--stages NUMS             Which stages to run (1-5, default: all)
--epochs NUMS             Epochs per stage (must match # of stages if set)
--learning-rate LR        Override learning rate
--early-stopping PAT      Early stopping patience (default: 6)
```

### Logging
```
-v, --verbose             Enable verbose console logging
--dashboard               Launch TensorBoard automatically
--dashboard-port PORT     TensorBoard port (default: 6006)
```

### Output
```
--output-dir PATH         Logs/checkpoints directory (default: experiments/logs)
--experiment-name NAME    Run identifier for organizing
```

### Export
```
--export-formats FMT      tflite/onnx/coreml (default: tflite)
--export-disabled         Skip export after training
--representative-batches  Batches for quantization (default: 100)
```

### Mobile Sync
```
--sync-mobile PATH        Mobile app root directory
--framework TYPE          flutter/react-native/swiftui (default: flutter)
--verify-sync             Verify models via SHA256 checksum
```

---

## Understanding Output

### Console Output (Normal Mode - WARNING+)
```
INFO     | M.A.L.L.I. Training Pipeline started
INFO     | Loading datasets...
INFO     | ✓ NIH dataset loaded
INFO     | ✓ Synthetic dataset loaded
```

### Console Output (Verbose Mode - INFO+)
```
INFO     | Logging initialized. Details written to: experiments/logs/logs/training_20250115_142345.log
INFO     | Loaded config from: configs/default.json
INFO     | Loading datasets...
INFO     | ✓ NIH dataset loaded
INFO     | ✓ Synthetic dataset loaded
INFO     | TensorBoard: tensorboard --logdir experiments/logs/tensorboard --port 6006
```

### Log File Output (DEBUG level)
```
2025-01-15 14:23:45 | __main__:156 | INFO     | Logging initialized
2025-01-15 14:23:47 | src.data.loaders.nih_loader:45 | DEBUG    | Loading dataset from path
2025-01-15 14:23:52 | __main__:268 | INFO     | ✓ Dataset loaded
```

---

## Directory Structure After Training

```
experiments/logs/
├── logs/
│   └── training_20250115_142345.log    ← Detailed debug logs
├── checkpoints/
│   ├── best_mobilenetv3_small.weights.h5
│   ├── last_mobilenetv3_small.keras
│   ├── stage_1_nih_warmup.weights.h5
│   ├── stage_2_nih_refine.weights.h5
│   ├── stage_3_synth_warmup.weights.h5
│   ├── stage_4_synth_refine.weights.h5
│   ├── stage_5_synth_polish.weights.h5
│   ├── decision_threshold.json         ← Calibrated threshold
│   └── exports/
│       ├── malaria_detector_tflite.tflite
│       ├── malaria_detector_onnx.onnx
│       ├── malaria_detector_coreml.mlmodel
│       └── metadata.json               ← Export info
└── tensorboard/
    └── 20250115-142345/                ← TensorBoard logs
        ├── train/
        ├── validation/
        └── [stage directories]
```

---

## Troubleshooting Commands

### Check TensorBoard manually
```bash
tensorboard --logdir experiments/logs/tensorboard --port 6006
# Then open: http://localhost:6006
```

### View latest log file
```bash
tail -f experiments/logs/logs/training_*.log
```

### Test imports only (no training)
```bash
python -c "from src.data.loaders.nih_loader import MalariaDataset; print('✓ Imports OK')"
```

### Verify export pipeline works
```bash
python train.py --stages 1 --epochs 1 --export-formats tflite --batch-size 64
```

### Clean up old runs
```bash
rm -rf experiments/logs/tensorboard experiments/logs/checkpoints
```

---

## Configuration File Format

Create `configs/custom.json`:
```json
{
  "data": {
    "batch_size": 32,
    "image_size": [224, 224]
  },
  "train": {
    "early_stopping_patience": 10
  },
  "export": {
    "representative_batches": 200
  }
}
```

Then use it:
```bash
python train.py --config configs/custom.json
```

---

## Performance Tips

1. **Faster Iteration**: Skip stages
   ```bash
   python train.py --stages 5 --epochs 5
   ```

2. **Faster Inference**: Increase batch size
   ```bash
   python train.py --batch-size 64
   ```

3. **Better Accuracy**: Lower learning rate + more patience
   ```bash
   python train.py --learning-rate 1e-5 --early-stopping 15
   ```

4. **Save Disk Space**: Disable export for dev
   ```bash
   python train.py --export-disabled
   ```

5. **Skip Synthetic Data**: Use only NIH
   ```bash
   python train.py --stages 1 2
   ```

---

## Integration with Mobile Apps

### Flutter
```bash
python train.py \
  --sync-mobile ../../apps/flutter_app \
  --framework flutter \
  --export-formats tflite
```
Models synced to: `flutter_app/assets/models/android/` and `flutter_app/assets/models/shared/`

### React Native  
```bash
python train.py \
  --sync-mobile ../../apps/rn_app \
  --framework react-native \
  --export-formats tflite
```

### SwiftUI
```bash
python train.py \
  --sync-mobile ../../apps/swiftui_app \
  --framework swiftui \
  --export-formats coreml
```

---

## Environment Setup

### First Time Only
```bash
# From ml/ directory
pip install -r requirements.txt

# Optional: Export support
pip install tf2onnx onnx onnxruntime      # For ONNX
pip install coremltools                    # For CoreML (macOS)
```

### Verify Setup
```bash
python -c "import tensorflow as tf; print(f'TF Version: {tf.__version__}')"
python train.py --help  # Should show all arguments
```

---

## Monitoring Training

### In Real-Time
```bash
# Terminal 1: Run training with dashboard
python train.py -v --dashboard

# Terminal 2: View logs (after training starts)
tail -f experiments/logs/logs/training_*.log

# Terminal 3: Monitor GPU (if available)
nvidia-smi -l 1  # Refresh every 1 second
```

### Post-Training
```bash
# View final metrics
grep "test results\|calibrated" experiments/logs/logs/training_*.log

# Check export status
cat experiments/logs/exports/metadata.json | python -m json.tool

# View mobile sync status
cat experiments/logs/exports/manifest.json | python -m json.tool
```

---

For detailed documentation, see: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
