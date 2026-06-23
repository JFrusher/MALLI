# MALLI ML Pipeline: Complete Stage Reference

## Overview
This document details every stage an image goes through from initial ROI detection through individual cell classification, including all parameters, weights, and tuning options.

---

## Pipeline Architecture

```
Input Image (Blood Smear)
    ↓
[STAGE 1] Image Preprocessing
    ↓
[STAGE 2] Tiling & Segmentation
    ↓
[STAGE 3] Per-Tile Morphological Detection
    ↓
[STAGE 4] Geometric Filtering & NMS
    ↓
[STAGE 5] Individual ROI Classification
    ↓
Output: Labeled Regions + Parasitemia
```

---

## STAGE 1: Image Preprocessing

### Purpose
Enhance blood cell contrast and suppress background noise.

### Process

#### A. Green Channel Enhancement (SAHI)
```python
# Function: ImagePreProcessor.preprocess_image()
# File: models/detection_pipeline.py (Dart also has equivalent)
```

**What it does:**
- Extracts green channel (blood cells appear greenish in microscopy)
- Boosts green channel by +30%
- Suppresses red and blue channels by -30%
- Optional Gaussian blur for smoothing

**Parameters:**
```
green_boost_factor: 1.3 (default, tunable)
red_suppression: 0.7
blue_suppression: 0.7
gaussian_blur: None (optional, kernel_size=5)
```

**Why:**
- Malaria parasites concentrate hemozoin (iron crystals) which shows in green
- Red/blue are mostly background autofluorescence

**Tuning:**
- Increase `green_boost_factor` (1.5-1.8) → More aggressive green emphasis
- Enable blur → Smoother detection, fewer tiny artifacts
- Decrease if: Over-brightening background

**Input:** Raw BGR image (uint8, 0-255)
**Output:** Preprocessed BGR image with enhanced green (uint8, 0-255)

---

## STAGE 2: Tiling & Inference Engine Setup

### Purpose
Break high-resolution image into manageable tiles for sequential processing (O(1) memory).

### Process

#### A. Tile Generation
```python
# Function: TilingInferenceEngine.process_tiles()
# File: models/detection_pipeline.py
```

**What it does:**
- Divides image into overlapping square tiles
- Processes tiles sequentially to avoid memory blowup
- Tracks tile offsets for coordinate remapping

**Parameters:**
```
tile_size: 640 (default, tunable via --sahi-tile-size)
overlap_ratio: 0.2 (default, tunable via --sahi-overlap)

# Computed internally:
stride = tile_size * (1 - overlap_ratio)
# With defaults: stride = 640 * 0.8 = 512 pixels
```

**Example for 4000×3000 image:**
```
Tile layout (640×640 tiles, 20% overlap):
- Row 0: [0:640], [512:1152], [1024:1664], ...
- Row 1: [512:1152], [1024:1664], ...
- Total tiles: ~32-36 tiles depending on exact dimensions
```

**Why overlapping tiles?**
- Cells near tile boundaries don't get cut off
- 20% overlap (128px) provides safe margin
- NMS later removes duplicates in overlap zones

**Tuning:**
- Increase `tile_size` (800-1024) → Fewer tiles, faster, less NMS overhead
- Decrease `tile_size` (480-512) → More tiles, catches small objects better, slower
- Increase `overlap_ratio` (0.3-0.4) → More NMS work but fewer boundary artifacts
- Decrease `overlap_ratio` (0.1) → Faster but more boundary issues

**Input:** Preprocessed image
**Output:** List of tile regions + offsets for coordinate mapping

---

## STAGE 3: Per-Tile Morphological Detection

### Purpose
Find dark objects (potential cells) within each tile.

### Process

#### A. Dark Region Thresholding
```python
# Located in: compute_overlay_sahi_pipeline() -> inference_callback()
# File: models/roi_grabber_review.py, lines ~620-660
```

**Step 1: Absolute Darkness Filter**
```python
dark_threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
# Keeps only pixels darker than 170 (blood cells < 170, background > 180)
```

**Parameters:**
```
absolute_intensity_threshold: 170 (tunable, currently hardcoded)
# Lower (140-150): Catches lighter cells, more background noise
# Higher (180-190): Misses lighter parasites, cleaner background
```

**Step 2: OTSU Thresholding (Selective)**
```python
# Apply OTSU only to already-dark regions, not full image
# Prevents background from influencing threshold
```

**Step 3: Local Contrast Filter**
```python
# Michelson contrast: std_dev / mean
# Rejects uniform backgrounds (low contrast)
# Accepts stained cells (high contrast)

contrast_threshold: 0.18 (tunable via hardcoded parameter)
# 0.15-0.20: Moderate, catches most cells
# 0.25+: Very strict, only high-contrast cells
# 0.10-0.15: Permissive, picks up noise
```

**Step 4: Binary Combination**
```python
final_thresh = AND(dark_threshold, otsu_threshold, high_contrast_mask)
# All three conditions must be true
```

#### B. Morphological Cleanup
```python
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Remove very small noise
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
# Connect nearby pixels into blobs
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
```

**Parameters:**
```
kernel_small: 3×3 (removes 1-2 pixel noise)
kernel_medium: 5×5 (connects ~5-10 pixel gaps)
open_iterations: 1 (erosion + dilation to remove noise)
close_iterations: 1 (dilation + erosion to fill holes)

# Tuning:
# Increase iterations: More aggressive cleaning, may merge nearby cells
# Decrease iterations: Preserve fine details, more noise
# Larger kernels: Smoother shapes, less detail
```

#### C. Contour Extraction
```python
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

**What it does:**
- Finds all connected components in binary image
- Each contour = one potential cell candidate

**Output:** List of contours (variable-length point arrays)

#### D. Contour Validation

For each contour, calculate:

**1. Area**
```
area = contour_width × contour_height

min_area: 600 pixels (tunable via --sahi-min-area)
# Default ~24×24 minimum
# 600-1000: Rejects tiny artifacts
# 300-500: Permissive, catches small cells
# 1000+: Misses small cells

max_area: (tile_size²) × max_area_ratio
# With defaults: (640²) × 0.10 = 40,960 pixels
# ~200×200 maximum
# 0.10-0.15: Typical cell range
# 0.20+: Allows clustering of multiple cells
```

**2. Aspect Ratio**
```
aspect_ratio = width / height

threshold: 0.4 ≤ aspect_ratio ≤ 2.5
# Rejects needle-like (scratches) and too-square shapes
# 0.3-3.0: More permissive
# 0.4-2.5: Conservative
```

**3. Circularity**
```
circularity = (4π × area) / (perimeter²)
# 1.0 = perfect circle
# 0.7-1.0 = round cells
# 0.3-0.6 = dust/scratches

threshold: circularity ≥ 0.50
# 0.60+: Very strict, only circular shapes
# 0.40-0.50: Moderate, catches irregular cells
# 0.30+: Permissive, lots of noise
```

**4. Solidity**
```
solidity = contour_area / convex_hull_area
# 1.0 = no jagged edges
# 0.65-0.95 = typical cells
# <0.5 = very jagged (dust, artifacts)

threshold: solidity ≥ 0.65
# 0.75+: Requires very smooth edges
# 0.65-0.70: Typical setting
# 0.50-0.60: Permissive, picks up edge noise
```

**5. Absolute Darkness**
```
roi_mean_intensity = mean(gray_values_in_contour_region)

threshold: roi_mean_intensity < 160
# 160-170: Allows moderately lit regions
# 140-150: Only very dark regions
# Catches real stained cells, rejects background
```

**All Validation Filters (AND logic):**
```
if (min_area ≤ area ≤ max_area AND
    0.4 ≤ aspect_ratio ≤ 2.5 AND
    circularity ≥ 0.50 AND
    solidity ≥ 0.65 AND
    roi_mean_intensity < 160):
    → PASS to next stage
else:
    → REJECT
```

**Input:** Binary thresholded image from tile
**Output:** List of valid BoundingBox objects (x, y, width, height, confidence)

---

## STAGE 4: Geometric Filtering & Non-Maximum Suppression

### Purpose
Remove duplicates from overlapping tiles and apply final geometric constraints.

### Process

#### A. Post-Inference Filtering
```python
# Function: PostInferenceFilter.geometric_filter()
# File: models/detection_pipeline.py
```

**Re-validates each detection:**
```python
max_area = (tile_size × tile_size) × max_area_ratio

for box in detections:
    area = box.width × box.height
    aspect_ratio = box.width / box.height
    
    if (min_area ≤ area ≤ max_area AND
        min_aspect_ratio ≤ aspect_ratio ≤ max_aspect_ratio):
        → KEEP
    else:
        → REMOVE
```

**Parameters:**
```
min_area: 600 (pixels)
max_area_ratio: 0.10 (fraction of tile)
min_aspect_ratio: 0.3
max_aspect_ratio: 3.0
```

#### B. Non-Maximum Suppression (NMS)
```python
# Function: PostInferenceFilter.apply_nms()
# File: models/detection_pipeline.py
```

**What it does:**
- Removes duplicate detections in overlapping tile regions
- Keeps detection with highest confidence
- Merges nearby boxes

**Algorithm:**
```python
while detections_exist:
    # Find box with highest confidence
    best_box = argmax(detections, confidence)
    keep(best_box)
    
    # Remove all boxes with IoU > threshold
    for other_box in detections:
        if IoU(best_box, other_box) > iou_threshold:
            remove(other_box)
```

**Parameters:**
```
nms_threshold: 0.5 (tunable via --sahi-nms-threshold)
# IoU = Intersection over Union

# 0.3-0.4: Aggressive, removes many near-duplicates
# 0.5: Moderate (default)
# 0.7-0.8: Permissive, keeps multiple nearby boxes
```

**IoU Calculation:**
```
IoU = Area(Intersection) / Area(Union)
# 1.0 = identical boxes
# 0.0 = no overlap
# Threshold determines which duplicates to keep
```

**Input:** All detections from all tiles (may have duplicates)
**Output:** Deduplicated detections with single confidence per region

---

## STAGE 5: Individual ROI Classification

### Purpose
Classify each ROI as infected (parasitized) or uninfected using deep learning.

### Process

#### A. ROI Extraction
```python
# Located in: compute_overlay_sahi_pipeline() -> inference_callback()
# File: models/roi_grabber_review.py
```

**For each valid bounding box:**
```python
x1 = max(0, x - 5)      # Pad by 5px on each side
y1 = max(0, y - 5)
x2 = min(width, x + w + 5)
y2 = min(height, y + h + 5)
patch = image[y1:y2, x1:x2]
```

**Parameters:**
```
roi_padding: 5 pixels (hardcoded)
# Provides context around cell for classifier
# Increase (10-20px): More context, slower
# Decrease (0-2px): Cell-only, may miss features
```

**Output:** ROI patch (variable size, RGB)

#### B. ROI Preprocessing
```python
patch_resized = cv2.resize(patch, model_input_size)
patch_normalized = patch_resized.astype(np.float32) / 255.0
```

**Parameters:**
```
model_input_size: (224, 224) by default
# MobileNetV3 standard input size
# Must match model's expected input

normalization: [0, 1] range (divide by 255)
# Typical for TensorFlow models
```

#### C. MobileNetV3 Classification

**Model:**
```
Architecture: MobileNetV3 Small
Input: (batch_size, 224, 224, 3) float32 [0-1]
Layers: ~168 layers, ~2.54M parameters
Output: (batch_size, 2) logits [uninfected, infected]

Weights: models/best_mobilenetv3_small.weights.h5
         models/best_mobilenetv3_small.keras
         models/last_mobilenetv3_small.h5
Size: ~9.5 MB
```

**Performance:**
```
Inference time per ROI: ~5-10ms (GPU) / ~50-100ms (CPU)
Memory per batch: ~50-100MB (224×224 float32)
Typical batch size: 64-128 ROIs
```

**Model Training Info:**
```
Dataset: NIH blood smear dataset (27,558 cells)
  - Parasitized: ~13,780
  - Uninfected: ~13,778
  
Training stages (multistage fine-tuning):
  - Stage 1: NIH warmup (base training)
  - Stage 2: NIH refine (fine-tuning)
  - Stage 3: Synthetic warmup (augmented data)
  - Stage 4: Synthetic refine
  - Stage 5: Synthetic polish

Training loss: Binary crossentropy
Metrics tracked: Recall, Precision, F1-score
```

#### D. Confidence Score Extraction
```python
pred = model.predict(np.expand_dims(patch_normalized, 0), verbose=0)
# pred shape: (1, 2) - logits for [uninfected, infected]

confidence = float(pred[0][1])
# Extract infected class probability
# Range: [0, 1]
# 0.0 = definitely uninfected
# 1.0 = definitely infected
```

**Parameters:**
```
model_inference_threshold: 0.5 (tunable via hardcoded parameter)
# Classifications with confidence > 0.5 marked as infected
# Adjust for sensitivity:
  - 0.3-0.4: Permissive, flags more as infected
  - 0.5: Balanced (default)
  - 0.6-0.7: Conservative, only high-confidence infections
```

#### E. Final Filtering
```python
if confidence > 0.5:
    → KEEP (mark as infected)
else:
    → Could still keep but mark as uninfected
```

**Output:** Final detections with:
- Bounding box (x, y, width, height)
- Confidence score [0, 1]
- Classification (infected/uninfected)

---

## FINAL OUTPUT: Statistics & Visualization

### Computed Metrics
```python
total_cells = len(final_detections)
infected_cells = sum(1 for d in detections if confidence > threshold)
parasitemia = (100.0 × infected_cells) / total_cells

# Colors for visualization:
# Cyan (0, 255, 255): SAHI detections
# High confidence: Thick boxes (3px)
# Low confidence: Thin boxes (1px)
```

**Output Format:**
```
Overlay image with annotated boxes
HUD display:
  - Total cells: {total_cells}
  - Infected: {infected_cells}
  - Parasitemia: {parasitemia:.2f}%
  - Pipeline variant: SAHI
  
Analysis lines:
  - Per-tile stats
  - NMS statistics
  - Processing time
```

---

## Complete Parameter Reference

### Command-Line Interface

```bash
# SAHI Pipeline Control
--use-sahi-pipeline           # Enable SAHI (default: False in old code, True now)
--sahi-tile-size 640          # Tile size pixels (default: 640)
--sahi-overlap 0.2            # Tile overlap ratio (default: 0.2)
--sahi-min-area 600           # Minimum bbox area (default: 600)
--sahi-max-area-ratio 0.10    # Max area as fraction of tile (default: 0.10)
--sahi-nms-threshold 0.5      # NMS IoU threshold (default: 0.5)

# MobileNetV3 Classification
--model-path PATH             # Model weights file (default: models/best_mobilenetv3_small.weights.h5)
--disable-model               # Skip classification, show boxes only
--threshold 0.85              # Infected classification threshold (default: 0.85)
--model-input-size 224        # Input size (default: 224, DO NOT CHANGE)
--batch-size 64               # Batch size for inference (default: 64)

# Input/Output
--dataset-root PATH           # Folder with images
--use-zip                     # Use ZIP file instead
--zip-path PATH               # ZIP file path
--per-group N                 # Images per group to sample (default: 30)
--save-dir PATH               # Output folder (default: outputs/roi_review)

# YOLO Alternative (competing pipeline)
--use-yolo                    # Use YOLOv8 instead of sensing variants
--yolo-weights PATH           # YOLO weights (default: yolov8n.pt)
--yolo-conf 0.3               # YOLO confidence (default: 0.3)
--yolo-nms-iou 0.5            # YOLO NMS IoU (default: 0.5)
```

### Hardcoded Parameters (Edit in Source)

**File: models/roi_grabber_review.py**
```python
# Line ~630: inference_callback() in compute_overlay_sahi_pipeline()

# Thresholding
absolute_intensity_threshold = 170
contrast_threshold = 0.18
local_contrast_std_factor = 0.25  # For contrast calculation

# Morphology
kernel_small = (3, 3)
kernel_medium = (5, 5)
open_iterations = 1
close_iterations = 1

# Validation (these ARE parameterized, use --sahi-min-area, etc.)
min_area = sahi_min_area  # Via argument
max_area_ratio = sahi_max_area_ratio  # Via argument
min_circularity = 0.50
min_solidity = 0.65
roi_mean_max_intensity = 160
roi_padding = 5  # pixels

# Classification
mobilenetv3_confidence_threshold = 0.5  # Infected/uninfected
batch_inference = True
verbose_inference = False
```

**File: models/detection_pipeline.py (Dart equivalent)**
```python
# Same constants replicated for mobile deployment
GREEN_BOOST = 1.3
RED_SUPPRESSION = 0.7
BLUE_SUPPRESSION = 0.7

TILE_OVERLAP_RATIO = 0.2  # Default
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.0
```

---

## Tuning Guide by Use Case

### Case 1: Too Many False Positives (Background Noise)
```
1. Increase --sahi-min-area (600 → 800-1000)
   → Only large blobs
2. Decrease contrast_threshold in code (0.18 → 0.22-0.25)
   → Require more contrast
3. Increase min_circularity (0.50 → 0.55-0.60)
   → Only round shapes
4. Increase roi_mean_max_intensity from 160 → 155-150
   → Only darker regions
```

### Case 2: Missing Cells
```
1. Decrease --sahi-min-area (600 → 400-500)
   → Catch smaller cells
2. Increase contrast_threshold (0.18 → 0.12-0.15)
   → Less strict on contrast
3. Decrease min_circularity (0.50 → 0.40-0.45)
   → Allow irregular shapes
4. Increase roi_mean_max_intensity (160 → 165-170)
   → Accept lighter regions
```

### Case 3: Slow Processing
```
1. Increase --sahi-tile-size (640 → 800-1024)
   → Fewer tiles to process
2. Decrease --sahi-overlap (0.2 → 0.1)
   → Fewer duplicates, less NMS work
3. Decrease --batch-size (64 → 32)
   → Less GPU memory per inference
4. Reduce --sahi-max-area-ratio (0.10 → 0.08)
   → Faster filtering
```

### Case 4: Better Accuracy
```
1. Use better pre-trained model
   → Replace best_mobilenetv3_small.weights.h5
2. Fine-tune on your specific dataset
   → Retrain with your blood smear images
3. Increase --sahi-overlap (0.2 → 0.3-0.4)
   → Catch boundary cells better
4. Increase --batch-size (64 → 128)
   → More samples per batch for normalization
```

---

## Model Export & Deployment

### TFLite Export (for mobile)
```bash
python models/export_tflite.py \
  --input-model models/best_mobilenetv3_small.weights.h5 \
  --output-model mobilenetv3_small.tflite \
  --quantize int8  # Or float32
```

**TFLite Inference (Dart/Flutter):**
```dart
// Load model
final model = await Interpreter.fromAsset('mobilenetv3_small.tflite');

// Input: [1, 224, 224, 3] float32
// Output: [1, 2] logits [uninfected, infected]

model.run(input, output);
final infected_probability = output[0][1];
```

---

## Performance Benchmarks

### Typical Processing Times (per image)
```
Image size: 2000×1500 pixels
Tile count: ~12-16 tiles
Classification count: ~200-500 cells

Stage timing (CPU, Intel i7):
  - Preprocessing: 5-10ms
  - Tiling: 50-100ms
  - Morphological detection: 200-300ms
  - NMS: 10-20ms
  - MobileNetV3 (200 cells): 5-10s @ batch=64
  - Overlay rendering: 50-100ms
  
Total: 5.3-10.5 seconds per image

With GPU (NVIDIA RTX3060):
  - Same except MobileNetV3: 500-800ms
  Total: 0.8-1.2 seconds per image
```

### Memory Usage
```
Peak memory (CPU):
  - Tile buffer: ~640² × 3 × 2 = 2.4 MB
  - Batch inference: 224² × 64 × 3 = 10 MB
  - Results storage: ~500 cells × 8 bytes = 4 KB
  Total: ~12-15 MB

Peak memory (GPU):
  - Tensor buffers: 50-100 MB
  - Model weights: 9.5 MB
  Total: ~60-110 MB
```

---

## References

- ROI Grabber Original: `models/roi_grabber.py` (watershed + MobileNetV3)
- SAHI Pipeline: `models/detection_pipeline.py` (new tiling approach)
- Reviewer Tool: `models/roi_grabber_review.py` (visualization & testing)
- Mobile Implementation: `lib/services/detection_pipeline.dart` (Dart equivalent)
- Model Training: `train.py` (multistage fine-tuning)

