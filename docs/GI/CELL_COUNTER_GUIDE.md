# Cell Counter Pipeline - Comprehensive Documentation

## Overview

The **CellCounter** is a production-ready pipeline for automated cell counting and malaria parasite detection in blood smear microscopy images. It combines image segmentation, region proposal extraction, and deep learning classification to provide accurate parasitemia measurements.

## Architecture

```
Input Image (BGR)
    ↓
[Preprocessing]        → CLAHE contrast normalization + Gaussian blur
    ↓
[Segmentation]         → Watershed (thick) or Morphological opening (thin)
    ↓
[ROI Extraction]       → Extract centered square bounding boxes from contours
    ↓
[Batch Prediction]     → MobileNetV3-Small binary classification
    ↓
[Threshold Filtering]  → Keep predictions ≥ 0.4 (infected)
    ↓
[NMS Deduplication]    → Remove overlapping detections (IOU > 0.5)
    ↓
[Output]               → Per-image cell counts and parasitemia statistics
```

## Installation

### Requirements
- TensorFlow 2.13.1
- OpenCV 4.x
- NumPy 1.24.3
- Python 3.8+

### Setup
```bash
# Install in the MALLI environment
pip install tensorflow==2.13.1 opencv-python numpy==1.24.3

# Import the module
from models import CellCounter, CellCountResult
```

## Quick Start

### Single Image Processing

```python
from models import CellCounter
import cv2

# Initialize (loads model once)
counter = CellCounter(verbose=True)

# Process image
result = counter.process_image("path/to/image.png")

# View results
print(result)  # Human-readable summary
print(result.to_dict())  # Dictionary for exporting

# Access details
print(f"Total cells: {result.total_cells}")
print(f"Infected: {result.infected_cells}")
print(f"Parasitemia: {result.parasitemia_percent:.2f}%")
print(f"Smear type: {result.smear_type}")
```

### Batch Processing

```python
# Process multiple images
image_paths = [
    "nih_data/cell_images/Infected/img1.png",
    "nih_data/cell_images/Uninfected/img2.png",
]

results = counter.process_batch(image_paths, skip_errors=True)

# Get statistics across batch
stats = counter.get_summary_statistics(results)
print(f"Mean parasitemia: {stats['mean_parasitemia_percent']:.2f}%")
```

### Directory Scanning

```python
# Process all images in a directory
results = counter.process_directory(
    "nih_data/cell_images",
    pattern="*.png",
    recursive=True,
)
```

## Command-Line Interface

### Basic Usage

```bash
# Single image
python -m models.cell_counter_example --image path/to/image.png

# Directory with recursive scan
python -m models.cell_counter_example \
  --directory nih_data/cell_images \
  --recursive
```

### Advanced Options

```bash
# Process with overlays and CSV export
python -m models.cell_counter_example \
  --directory nih_data \
  --recursive \
  --output-csv results.csv \
  --output-overlays overlays/ \
  --roi-size 128 \
  --batch-size 64 \
  --nms-iou 0.5 \
  --verbose
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` | Path | - | Process single image |
| `--directory` | Path | - | Process directory |
| `--pattern` | str | `*.png` | Glob pattern for images |
| `--recursive` | flag | False | Search subdirectories |
| `--smear-type` | str | `auto` | Force `thin`/`thick` or `auto`-detect |
| `--weights` | Path | `models/best_mobilenetv3_small.weights.h5` | Model weights |
| `--threshold` | Path | `models/decision_threshold.json` | Threshold JSON |
| `--roi-size` | int | 128 | ROI crop size (pixels) |
| `--batch-size` | int | 32 | Model batch size |
| `--nms-iou` | float | 0.5 | NMS IOU threshold |
| `--output-csv` | Path | - | Export results to CSV |
| `--output-overlays` | Path | - | Save overlay images |
| `--verbose` | flag | False | Debug logging |

## API Reference

### CellCounter Class

Main interface for the counting pipeline.

#### Initialization

```python
counter = CellCounter(
    weights_path="models/best_mobilenetv3_small.weights.h5",
    threshold_path="models/decision_threshold.json",
    model_input_size=(224, 224),
    roi_size=128,
    batch_size=32,
    nms_iou_threshold=0.5,
    min_blob_area=80.0,
    max_blob_area=50_000.0,
    verbose=False,
)
```

**Parameters:**
- `weights_path`: Path to MobileNetV3 weights (h5 or keras format)
- `threshold_path`: Path to `decision_threshold.json`; if None, uses 0.4
- `model_input_size`: MobileNetV3 input resolution (H, W)
- `roi_size`: Size of square ROI crops around cell centroids
- `batch_size`: Batch size for model inference
- `nms_iou_threshold`: IOU threshold for NMS (0.0-1.0)
- `min_blob_area`: Minimum cell area in pixels (filters noise)
- `max_blob_area`: Maximum cell area in pixels (filters artifacts)
- `verbose`: Enable DEBUG logging

#### Methods

##### `process_image(image_path, smear_type="auto")`

Process a single image and return cell counts.

**Returns:** `CellCountResult`

```python
result = counter.process_image("image.png", smear_type="auto")
```

**Raises:**
- `FileNotFoundError`: If image doesn't exist
- `ValueError`: If image can't be read
- `RuntimeError`: If processing fails

##### `process_batch(image_paths, smear_type="auto", skip_errors=True)`

Process multiple images efficiently.

**Returns:** `list[CellCountResult]`

```python
results = counter.process_batch(image_paths, skip_errors=True)
```

##### `process_directory(directory, pattern="*.png", recursive=True, smear_type="auto", skip_errors=True)`

Scan directory and process matching images.

**Returns:** `list[CellCountResult]`

```python
results = counter.process_directory("nih_data", pattern="*.png", recursive=True)
```

##### `get_summary_statistics(results)`

Compute aggregated statistics across multiple results.

**Returns:** `dict` with keys:
- `num_images`: Number of processed images
- `total_cells`: Total cells detected
- `total_infected`: Total infected cells
- `total_uninfected`: Total uninfected cells
- `mean_parasitemia_percent`: Mean parasitemia across images
- `std_parasitemia_percent`: Standard deviation
- `min_parasitemia_percent`: Minimum parasitemia
- `max_parasitemia_percent`: Maximum parasitemia
- `smear_type_distribution`: Dict with count per smear type

```python
stats = counter.get_summary_statistics(results)
print(f"Mean parasitemia: {stats['mean_parasitemia_percent']:.2f}%")
```

### CellCountResult Dataclass

Holds results from processing a single image.

**Attributes:**
- `image_path: str` - Path to the processed image
- `total_cells: int` - Total cells detected
- `infected_cells: int` - Cells above threshold
- `uninfected_cells: int` - Cells below threshold
- `parasitemia_percent: float` - Percentage infected
- `raw_proposal_count: int` - Proposals before NMS
- `smear_type: str` - "thin" or "thick"
- `proposals: Sequence[RoiProposal]` - Extracted ROIs
- `probabilities: np.ndarray` - Model outputs [0, 1]
- `kept_indices: list[int]` - Indices of infected cells after NMS
- `metadata: dict` - Additional statistics

**Methods:**
- `__str__()` - Human-readable summary
- `to_dict()` - Dictionary for CSV/JSON export

```python
result = results[0]
print(result)  # "image.png (thin): 25/100 infected (25.0%)"
d = result.to_dict()  # Export-ready dict
```

## Processing Pipeline Details

### 1. Preprocessing

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the green channel:
- Clip limit: 2.0
- Tile grid: 8×8
- Gaussian blur: kernel 5×5

Enhances local contrast while preserving edges.

### 2. Segmentation

**Thick Smears:** Watershed algorithm
- Morphological opening (iterations=2)
- Distance transform (DIST_L2)
- Marker-based watershed
- Connected component analysis

**Thin Smears:** Morphological operations
- Inverse binary threshold (Otsu)
- Morphological opening (iterations=1)
- Median blur (kernel 3×3)

### 3. ROI Extraction

For each connected component in the binary mask:
1. Compute contour area (filters: 80 to 50,000 pixels)
2. Calculate centroid using image moments
3. Extract centered square box (128×128 default)
4. Clip to image boundaries

### 4. Batch Prediction

MobileNetV3-Small binary classifier:
- Input: 224×224 RGB
- Output: Sigmoid probability [0, 1]
- 1 = Infected (parasite present)
- 0 = Uninfected (healthy cell)

### 5. Threshold Filtering

Keep detections where:
```
probability >= threshold  (default: 0.4)
```

Calibrated threshold from `decision_threshold.json`:
- Optimizes F2-score (recall-focused)
- Minimizes false negatives
- ~96% recall at ~67% precision

### 6. Non-Maximum Suppression

De-duplicates overlapping infected detections:
- Computes IOU between positive predictions
- Sorts by confidence
- Greedy suppression (IOU > 0.5)
- Keeps high-confidence detections

## Output Formats

### CSV Export

```python
from models.cell_counter_example import export_results_csv

export_results_csv(results, Path("results.csv"))
```

**Columns:**
- `image_path`: Path to image
- `smear_type`: "thin" or "thick"
- `total_cells`: Total detected cells
- `infected_cells`: Cells above threshold
- `uninfected_cells`: Cells below threshold
- `parasitemia_percent`: Infection percentage
- `raw_proposal_count`: Proposals before NMS

### Overlay Images

Visualization with bounding boxes:
```python
from models.cell_counter_example import save_overlay

for result in results:
    image_bgr = cv2.imread(result.image_path)
    save_overlay(image_bgr, result, output_path)
```

**Colors:**
- **Red boxes** (thickness=2): Infected cells (above threshold, kept after NMS)
- **Green boxes** (thickness=1): Uninfected cells (below threshold)
- **Labels**: Confidence scores

## Configuration

### Tuning Parameters

| Parameter | Range | Impact |
|-----------|-------|--------|
| `threshold` | 0.0-1.0 | Sensitivity; higher = fewer detections |
| `roi_size` | 64-256 | Cell crop size; default 128 |
| `min_blob_area` | 10-200 | Filters small noise artifacts |
| `max_blob_area` | 1000-100k | Filters large background regions |
| `nms_iou_threshold` | 0.0-1.0 | De-duplication; higher = more lenient |
| `batch_size` | 16-128 | Memory/speed tradeoff |

### Recommended Configurations

**Sensitivity (maximize detection):**
```python
counter = CellCounter(
    threshold=0.3,
    min_blob_area=50,
    nms_iou_threshold=0.3,
)
```

**Specificity (maximize accuracy):**
```python
counter = CellCounter(
    threshold=0.5,
    min_blob_area=100,
    nms_iou_threshold=0.7,
)
```

**Speed (fast batch processing):**
```python
counter = CellCounter(
    batch_size=64,
    roi_size=112,  # Slightly smaller
    nms_iou_threshold=0.3,
)
```

## Performance

### Model
- **Architecture:** MobileNetV3-Small
- **Input:** 224×224 RGB
- **Output:** Binary sigmoid
- **Size:** ~2.5 MB
- **Inference:** ~0.5-1.0ms per ROI (GPU), ~5ms (CPU)

### Accuracy (on NIH dataset)
- **Recall:** 96.1% (catches almost all infected cells)
- **Precision:** 66.7% (some false positives)
- **F2-Score:** 0.883 (recall-optimized)
- **Balanced Accuracy:** 83.1%

### Speed
- **Single image:** 1-5 seconds (depends on cell count)
- **Batch (100 images):** 2-10 minutes (GPU)

## Troubleshooting

### No cells detected

**Causes:**
- Image quality too poor
- Smear type detection incorrect
- Segmentation parameters not suitable

**Solutions:**
```python
# Force smear type
result = counter.process_image(image, smear_type="thick")

# Lower detection threshold (more sensitive)
counter = CellCounter(min_blob_area=40)

# Check intermediate segmentation
mask = segment_thick_smear_watershed(image_bgr, preprocessed)
cv2.imshow("Mask", mask)
```

### Too many false positives

**Solutions:**
```python
# Raise classification threshold
counter = CellCounter(threshold=0.5)

# Stricter NMS
counter = CellCounter(nms_iou_threshold=0.7)

# Filter larger cells
counter = CellCounter(max_blob_area=30_000)
```

### Memory issues with large images

**Solutions:**
```python
# Reduce batch size
counter = CellCounter(batch_size=16)

# Reduce ROI size (model still works at 112×112)
counter = CellCounter(
    roi_size=96,
    model_input_size=(112, 112),  # Smaller model input
)
```

## Integration Examples

### With Jupyter Notebook

```python
from models import CellCounter
import matplotlib.pyplot as plt

counter = CellCounter()
result = counter.process_image("image.png")

# Plot statistics
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(["Infected", "Uninfected"], [result.infected_cells, result.uninfected_cells])
ax1.set_title("Cell Counts")
ax2.hist(result.probabilities, bins=20)
ax2.set_title("Classification Scores")
plt.show()
```

### With Database

```python
import sqlite3

results = counter.process_batch(image_paths)

# Save to SQLite
conn = sqlite3.connect("results.db")
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS cells (
    id INTEGER PRIMARY KEY,
    image_path TEXT,
    total_cells INTEGER,
    infected_cells INTEGER,
    parasitemia REAL
)""")

for r in results:
    cursor.execute(
        "INSERT INTO cells VALUES (NULL, ?, ?, ?, ?)",
        (r.image_path, r.total_cells, r.infected_cells, r.parasitemia_percent),
    )
conn.commit()
```

## Development

### Adding Custom Segmentation

```python
from models.cell_counter import CellCounter
from models.roi_grabber import extract_roi_proposals, batch_predict_rois

class CustomCellCounter(CellCounter):
    def process_image(self, image_path, smear_type="auto"):
        # ... load image ...
        
        # Custom segmentation
        mask = my_custom_segmentation(image_bgr)
        
        # Use rest of pipeline
        proposals = extract_roi_proposals(mask, image_bgr.shape)
        # ... etc ...
```

### Custom Thresholding

```python
# Use different threshold per image
for image_path in image_paths:
    result = counter.process_image(image_path)
    
    # Re-threshold post-hoc
    if result.parasitemia_percent > 50:
        # High parasitemia - use stricter threshold
        high_confidence = [
            i for i in result.kept_indices
            if result.probabilities[i] > 0.7
        ]
```

## References

- **Model:** MobileNetV3-Small (Transfer learning from ImageNet)
- **Segmentation:** OpenCV watershed and morphological operations
- **Dataset:** NIH Malaria Microscopy Dataset (blood smear images)
- **Paper:** NIH malaria dataset paper reference

## Support

For issues or questions:
1. Check troubleshooting section
2. Enable `verbose=True` for debug logs
3. Verify model files and paths
4. Check input image format and dimensions
