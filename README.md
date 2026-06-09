# MALLI

[![Version](https://img.shields.io/badge/version-0.0.1-blue)](pubspec.yaml)
[![License](https://img.shields.io/badge/license-LICENSE-lightgrey)](LICENSE)

MALLI is a blood-smear malaria analysis project that combines a TensorFlow training and evaluation pipeline with a Flutter mobile app for offline-friendly sample capture and result review.

## What the project does

The repository contains two connected parts:

- A Python ML stack for loading malaria datasets, training a MobileNetV3-small classifier, evaluating saved weights, and exporting INT8 TFLite models.
- A Flutter app that displays completed samples, captures new images, and stores results locally with SQLite.

The main Python entry points are [train.py](train.py), [models/evaluate.py](models/evaluate.py), and the helpers under [models/](models/). The Flutter app starts from [lib/main.dart](lib/main.dart).

## Why the project is useful

MALLI is designed for field and low-resource workflows where malaria screening needs to be portable, repeatable, and easy to review. It provides:

- A staged training pipeline for NIH and synthetic blood-smear data.
- TFRecord caching for faster dataset reuse on larger runs.
- Quantized model export for mobile deployment.
- A cell-counting pipeline for parasite and ROI analysis.
- A simple Flutter UI for reviewing completed samples on device.

## How to get started

### Prerequisites

- Python 3.10 or newer
- Conda or `venv`
- Flutter SDK
- Access to the malaria datasets used by the training scripts

### Python setup

Using Conda:

```bash
conda env create -f environment.yml
conda activate malli
pip install -r requirements.txt
```

Using `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The pinned Python stack is intentional. The checked-in versions in [requirements.txt](requirements.txt) and [environment.yml](environment.yml) are the safest place to start.

### Flutter setup

```bash
flutter pub get
```

### Train a model

```bash
python train.py --launch-dashboard
```

This runs the staged training pipeline defined in [train.py](train.py). You can pass `--config` to load a JSON override file.

### Evaluate saved weights

```bash
python models/evaluate.py \
  --weights models/best_mobilenetv3_small.weights.h5 \
  --data-root nih_data \
  --synthetic-root synthetic_field_ready \
  --synthetic-labels-csv labels.csv
```

### Run the Flutter app

```bash
flutter run
```

### Quick Python usage

```python
from models import predict_image, CellCounter

probability, label = predict_image(
    "models/best_mobilenetv3_small.weights.h5",
    "nih_data/cell_images/Parasitized/example.png",
)

counter = CellCounter(verbose=True)
result = counter.process_image("nih_data/cell_images/Parasitized/example.png")
print(probability, label)
print(result.to_dict())
```

## Where to get help

Start with the project docs that describe the supported workflows:

- [START_HERE_TRAINING_PIPELINE.md](START_HERE_TRAINING_PIPELINE.md)
- [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)
- [docs/CELL_COUNTER_GUIDE.md](docs/CELL_COUNTER_GUIDE.md)
- [docs/PIPELINE_STAGES_REFERENCE.md](docs/PIPELINE_STAGES_REFERENCE.md)
- [docs/Roadmap.md](docs/Roadmap.md)
- [docs/deep_dive/README.md](docs/deep_dive/README.md)
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- [DETECTION_PIPELINE_IMPLEMENTATION.md](DETECTION_PIPELINE_IMPLEMENTATION.md)

If you are debugging a specific module, the most useful starting points are [data/data_loader.py](data/data_loader.py), [data/synthetic_data_loader.py](data/synthetic_data_loader.py), [models/inference.py](models/inference.py), and [lib/services/image_processor.dart](lib/services/image_processor.dart).

## Who maintains and contributes

The project is maintained in the `JFrusher/MALLI` repository workspace by the MALLI project team.

Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes focused, document behavior changes, and run the relevant Python or Flutter checks before opening a pull request.

If you are extending the ML pipeline, start from [train.py](train.py) and the files under [models/](models/). For mobile work, start from [lib/main.dart](lib/main.dart) and the files under [lib/](lib/).
