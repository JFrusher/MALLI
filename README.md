# M.A.L.L.I. — Mobile Advanced Lightweight Localization & Imaging

> Lightweight, mobile-first malaria microscopy classifier and synthetic-data pipeline.

## What this project does

M.A.L.L.I. trains and exports a compact malaria cell classifier (MobileNetV3-based) that is robust to low-cost Foldscope-style optics and thick-smear preparation. The repository includes:

- Staged training pipeline with curriculum-style augmentations (`train.py`).
- Synthetic data generator that produces field-ready images and soft labels (`utils/synthetic_field_ready_dataset.py`, `synthetic_field_ready/`).
- Model factory, training checkpoints, evaluation and export utilities (`models/`).
- Live dashboard and TensorBoard integration (`utils/dashboard.py`, `logs/tensorboard/`).

## Why this is useful

- Produces lightweight INT8-ready models suitable for on-device inference.
- Uses synthetic data and optical simulation to bridge lab-to-field performance gap.
- Curriculum augmentation and soft-labeling improve robustness and confidence calibration.

## Quick Start

Requirements (see `requirements.txt`):

```bash
python 3.10+
pip install -r requirements.txt
```

Prepare data:

- Dataverse ZIP: place `dataverse_files.zip` in the project root, or point the config at its location.
- First run: the loader will build compressed TFRecord shards under `datasets/blood_smear/processed/tfrecord/` so later training runs stream without reopening the ZIP.
- NIH dataset: if you already extracted the archive, place the cell images under `nih_data/cell_images/Parasitized` and `nih_data/cell_images/Uninfected`.
- (Optional) pre-generate synthetic field-ready images into `synthetic_field_ready/` using the utilities in `utils/`.

Training (default config):

```bash
python train.py --config path/to/config.json
# or to launch tensorboard automatically while training
python train.py --launch-dashboard
```

Key config knobs are embedded in `train.py`'s `DEFAULT_CONFIG` (dataset paths, stages, export settings, and dashboard options).
The default data path now prefers cached TFRecord streaming and uses `dataverse_files.zip` as the source archive.

Generate synthetic dataset (example; see `docs/synthetic_data_production.md` for full options):

```bash
python -m utils.synthetic_field_ready_dataset \
  --dataset-root nih_data \
  --output-dir synthetic_field_ready \
  --num-samples 10000 \
  --seed 42
```

Exported model and inference

- Trained checkpoints and exported artifacts live in `models/` (e.g. `mobilenetv3_small_int8.tflite`, `.h5`, `.keras`).
- Quick inference example:

```python
from models.inference import load_model, predict_image

model = load_model('models/mobilenetv3_small_int8.tflite')
result = predict_image(model, 'path/to/image.png')
print(result)
```

See `models/export_tflite.py` and `models/inference.py` for full export and runtime examples.

## Project structure (high level)

- `train.py` — main staged training pipeline and curriculum config
- `data/` — dataset loaders (`data_loader.py`, `synthetic_data_loader.py`)
- `utils/` — helpers and synthetic dataset generator (`synthetic_field_ready_dataset.py`, `dashboard.py`)
- `models/` — model factory, evaluation, export, inference helpers
- `synthetic_field_ready/` — generated synthetic images and `labels.csv` (optional)
- `docs/` — design docs, roadmap, and synthetic-data production notes

## Getting help & documentation

- Read design and usage details in `docs/synthetic_data_production.md` and `docs/Roadmap.md`.
- For issues or questions, open a GitHub Issue in this repository.
- For TensorBoard logs and live metrics, point TensorBoard at `logs/tensorboard/`:

```bash
tensorboard --logdir logs/tensorboard --port 6006
```

## Maintainers & Contribution

- Maintained by the project authors. To contribute:
  - Fork the repo and open a pull request against `main`.
  - Open issues for bugs, feature requests, or dataset/benchmark contributions.

- For contribution guidelines, see `docs/` or add `docs/CONTRIBUTING.md` and link it here.

## Notes & best practices

- Use the staged-training defaults in `train.py` for reproducible experiments.
- When exporting for mobile, validate INT8 models on a representative calibration set (see `train.py` export settings).
- Keep training logs under `logs/` and models under `models/` (both are already gitignored).

## License

See the repository `LICENSE` file for license terms.

---

If you'd like, I can: (a) add a minimal `docs/CONTRIBUTING.md`, (b) insert example `config.json` templates, or (c) add CI badges and a changelog entry. Which would you prefer next?
