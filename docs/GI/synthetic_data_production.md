# Synthetic Data Production Process for M.A.L.L.I.

## Executive Summary

The M.A.L.L.I. project generates synthetic malaria microscopy images from real NIH cell samples to create a **field-ready training dataset** that simulates the constraints and artifacts of portable foldscope imaging in low-resource settings. This document provides full transparency into the methodology, justification, and implementation of this augmentation pipeline.

---

## Problem Statement

Malaria detection models trained on high-quality laboratory microscopy images often fail in the field due to:

1. **Optical degradation**: Foldscope imaging introduces chromatic aberration, focus blur, vignetting, and dust/bubble artifacts
2. **Thick-smear preparation**: Field samples contain overlapping cells with depth variation and partial occlusion
3. **Staining variability**: Giemsa stain application is inconsistent across field locations and operators
4. **Limited diversity**: Publicly available datasets (NIH malaria dataset) consist of individual, isolated cells

By generating synthetic data that **replicates these real-world challenges**, we train more robust models that generalize better to actual field deployments.

---

## Data Generation Pipeline

### Step 1: Source Data Collection

**Input**: NIH Malaria Dataset (cell_images/)
- **Parasitized folder**: ~24,955 images of infected red blood cells
- **Uninfected folder**: ~24,955 images of uninfected red blood cells
- **Format**: Grayscale microscopy images (~150×150 px), standardized acquisition

**Loading Strategy**:
```python
ThickSmearGenerator(dataset_root, image_size=(224, 224), seed=42)
```

The generator:
1. Discovers source folders (supports multiple search paths for flexibility)
2. Recursively collects all image files from `Parasitized/` and `Uninfected/` subdirectories
3. Sorts paths for deterministic ordering
4. Validates that both classes exist before proceeding

Each image is **loaded with OpenCV** and converted to RGB format (standardized to 224×224 for model input).

---

### Step 2: Image Generation Modes

The pipeline generates synthetic samples using three distinct modes. Each mode targets a specific field condition and is selected probabilistically during training:

#### **Mode 1: Stacking (45% probability)**

**Purpose**: Simulate thick-smear preparation and depth/bunching artifacts.

**Implementation**:
```python
def stacking(self) -> Tuple[np.ndarray, float, List[str]]:
    num_cells = random(2, 4)  # Randomly select 2-4 cells
    images = load_and_resize(random_paths(num_cells))
    
    # Alpha-blend with graduated transparency
    alphas = [1.0, random(0.40, 0.70), random(0.40, 0.70), ...]
    blended = _torch_blend(images, alphas)
    soft_label = blend_labels(labels, alphas)
```

**Why This Works**:
- **Thick smears** contain 2-4 overlapping cells per observation area
- **Alpha blending** with graduated transparency (first cell opaque, subsequent layers 40-70% transparent) creates realistic depth perception
- **PyTorch tensor compositing** ensures numerically stable floating-point blending: `result = img[0] * 1.0 + img[1] * α₁ + img[2] * α₂ ...`
- **Soft labels** are computed using the same alpha weights, creating continuous labels that reflect the true mixing ratio

**Example**: If we blend 3 cells [Uninfected, Parasitized, Uninfected] with alphas [1.0, 0.55, 0.62]:
- Visual result: A mixed appearance with infection signal from the middle layer
- Soft label: 0.0 × 1.0 + 1.0 × 0.55 + 0.0 × 0.62 = **0.55** (soft-infected label, not binary)

---

#### **Mode 2: Custom CutMix (30% probability)**

**Purpose**: Simulate partial occlusion and high-density smear regions where infected parasite fragments cover clean cells.

**Implementation**:
```python
def custom_cutmix(self) -> Tuple[np.ndarray, float, List[str]]:
    clean_image = load(random_uninfected_path())
    infected_image = load(random_parasitized_path())
    
    # Create irregular mask covering 10-30% of the image
    mask = _fragment_mask(image.shape[:2])
    
    # Blend: keep clean areas, overlay infected in masked regions
    mixed = clean × (1 - mask) + infected × mask
    soft_label = mask.mean()  # Coverage fraction as soft label
```

**Mask Generation**:
- Generate 2-4 random ellipses with:
  - Random center position
  - Random radius (1/16 to 1/5 of image dimensions)
  - Random rotation angle (0-180°)
- Apply Gaussian blur (σ=4.0) to create soft edges (no hard boundaries)
- Normalize mask to [0, 1] with soft thresholding
- Ensure target coverage (10-30%) is achieved

**Why This Works**:
- **Realistic partial occlusion**: In high-density smears, parasites are partially obscured by overlapping cells
- **Irregular, natural shapes**: Ellipses are more realistic than rectangles; Gaussian blur prevents artificial hard edges
- **Soft labels track coverage**: The pixel-level mean of the mask directly measures infection contribution, enabling continuous labels
- **Single-pair mixing**: Unlike stacking, this uses exactly one clean + one infected cell, isolating the occlusion signal

**Example**: If 18% of pixels are masked and overlaid with infected material:
- Visual result: Clean cell with infected regions inserted
- Soft label: **0.18** (18% infection signal)

---

#### **Mode 3: Clean Sample (25% probability)**

**Purpose**: Maintain diversity by including individual cells; allows the model to learn baseline characteristics without augmentation interaction effects.

**Implementation**:
```python
def clean_sample(self) -> Tuple[np.ndarray, float, List[str]]:
    cell = load(random_cell_path())
    label = infer_label(path)  # 0 = Uninfected, 1 = Parasitized
    return cell, float(label), [path]
```

**Why This Works**:
- Prevents overfitting to blended/occluded patterns
- Balances the curriculum: early epochs use mostly clean samples, later epochs emphasize complex modes
- Individual cells have deterministic labels (0 or 1)

---

### Step 3: Foldscope Optics Simulation

**After** generation mode is applied, **all images** pass through the foldscope pipeline to simulate field-deployment optical constraints.

#### 3A: Chromatic Aberration Shift
**Mimics**: Lateral misalignment of red and blue channels (common in low-cost lens assemblies).

**Implementation**:
```python
transform_red = [[1, 0, dx_red],      # 2×3 affine matrix
                  [0, 1, dy_red]]
transform_blue = [[1, 0, dx_blue],    # Shift blue channel independently
                   [0, 1, dy_blue]]

red_channel = cv2.warpAffine(img[:,:,0], transform_red, ...)
blue_channel = cv2.warpAffine(img[:,:,2], transform_blue, ...)
result = stack([red, green, blue])
```

**Hyperparameters**:
- Shift range: ±2 to ±5 pixels
- Probability: 85%

**Why**: Foldscope optics are built from stacked lenses; low-cost assembly introduces misalignment. This shift is visible as color fringing at cell boundaries—a real artifact that models must handle.

---

#### 3B: Vignetting (Peripheral Darkening)
**Mimics**: Radial light falloff toward image edges (natural in wide-angle, short-focal-length systems).

**Implementation**:
```python
# Radial distance from image center (normalized)
radius = sqrt((y - center_y)² + (x - center_x)²)

# Multiplicative mask: 1.0 at center, 0.25 at corners
mask = clip(1.0 - strength × radius², 0.25, 1.0)
vignetted = image × mask
```

**Hyperparameters**:
- Strength: 30-60% dimming
- Probability: 90%

**Why**: Foldscope's wide field of view and short working distance create strong vignetting. Models must learn that peripheral darkening is optical, not pathological.

---

#### 3C: Focus Blur (OneOf)
**Mimics**: Out-of-focus regions, motion blur, and glass artifacts common in field imaging.

**Implemented as a random choice**:
1. **Gaussian blur**: σ ∈ [1.5, 3.5], kernel ∈ [7, 13]
2. **Motion blur**: Directional blur (13-pixel kernel)
3. **Glass blur**: Crystalline distortion (σ=0.9, Δ=5, 2 iterations)

**Probability**: 95%

**Why**: Foldscope field deployment often has focus inconsistency (manual focus, vibration, thick samples). Motion blur represents jittery capture. Glass blur simulates bubbles/dust particles acting as micro-lenses.

---

#### 3D: Dust and Bubble Artifacts
**Mimics**: Physical occlusions from dust particles, air bubbles, and slide defects.

**Implementation**:
```python
# Dust: 1-3 small circles (2-6 px radius) at intensity 40-100
dust_layer = create_circles(count=1-3, radius=2-6px, intensity=40-100)
dust_layer = blur(dust_layer, σ=1.5)  # Soft edges
image *= (1 - dust_layer)  # Darken under dust

# Bubble (5% chance): Large circle (12-32 px) with soft Gaussian falloff
bubble_mask = blur(large_circle(...), σ=2.0)
image *= (1 - bubble_mask) + 255 * bubble_mask  # Whiten bubble region
```

**Why**: Foldscope slides are often prepared hastily in field settings, resulting in:
- Dust particles on lens/slide
- Air bubbles trapped in mounting medium
- These are realistic defects models **must** be robust to

---

#### 3E: Stain Jitter (Hue and Saturation Variation)
**Mimics**: Inconsistent Giemsa staining across field locations and time.

**Implementation**:
```python
# Convert RGB → HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Perturb hue: ±27 steps in OpenCV hue space [0, 180]
# This represents ±15% of the hue spectrum
hue_shift = random(-27, 28)
hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

# Saturation scale: 0.8x to 1.25x
saturation_scale = random(0.80, 1.25)
hsv[:, :, 1] = clip(hsv[:, :, 1] * saturation_scale, 0, 255)

result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```

**Why**: Staining is manual in the field. Temperature, dye age, pH, and water quality all affect color. We simulate this with continuous shifts in hue (color tone) and saturation (color intensity). This is critical for models to learn **parasite morphology**, not just red-brown color.

---

### Step 4: Curriculum Augmentation Scheduling

**Purpose**: **Ease the model into complexity** rather than overwhelming it with full augmentation from epoch 1.

**Implementation**:
```python
class CurriculumAugmentor:
    def update_epoch(self, epoch: int) -> float:
        progress = (epoch - 1) / (max_epoch - 1)  # [0, 1]
        for transform in [chromatic, vignette, blur, dust, stain]:
            transform.p = base_probability × progress
        return progress
```

**Timeline Example (40 epochs)**:
- **Epoch 1**: All transforms at ~0% probability → mostly clean NIH images
- **Epoch 20**: All transforms at ~50% probability → moderate field artifacts
- **Epoch 40**: All transforms at ~100% probability → full foldscope simulation

**Why**:
- Models learn clean cell morphology first (stable feature learning)
- Gradually adapt to artifacts as they build capacity
- Prevents early training collapse from overwhelming noise
- Allows transfer from synthetic → real data without catastrophic forgetting

---

## Label Generation Strategy

### Soft Labels

Unlike hard binary labels (0 or 1), **soft labels** reflect the true mixing/occlusion ratio:

| Generation Mode | Label Computation |
|---|---|
| **Stacking** | Weighted average of source labels using alpha blend weights |
| **CutMix** | Mean pixel value of the occlusion mask |
| **Clean** | Direct label from source path (0 or 1) |

**Example**:
```
Stacking: [Uninfected, Parasitized, Uninfected] with alphas [1.0, 0.60, 0.55]
soft_label = 0 × 1.0 + 1 × 0.60 + 0 × 0.55 = 0.60
hard_label = int(0.60 ≥ 0.5) = 1
```

**Benefit**: Models trained with soft labels learn **confidence calibration**. A label of 0.60 means "moderately infected," not "definitely infected." This improves uncertainty quantification on real field data.

---

## Output Format

### Directory Structure
```
synthetic_field_ready/
├── labels.csv
├── synthetic_000000.png
├── synthetic_000001.png
├── ...
└── synthetic_NNNNNN.png
```

### CSV Schema
```csv
filename,soft_label,hard_label,operation,source_paths
synthetic_000000.png,0.625000,1,stack,path/to/Parasitized/img1.png|path/to/Uninfected/img2.png|path/to/Parasitized/img3.png
synthetic_000001.png,0.155000,0,cutmix,path/to/Uninfected/base.png|path/to/Parasitized/overlay.png
synthetic_000002.png,1.000000,1,clean,path/to/Parasitized/clean.png
```

**Fields**:
- `filename`: PNG file name (deterministic ordering for reproducibility)
- `soft_label`: Floating-point infection probability [0, 1]
- `hard_label`: Binary classification label (soft_label ≥ 0.5)
- `operation`: Generation mode (stack, cutmix, clean)
- `source_paths`: Pipe-delimited list of source images (for provenance/debugging)

---

## Reproducibility & Accountability

### Seeds and Determinism

```python
generator = ThickSmearGenerator(
    dataset_root="nih_data",
    seed=42  # Fixed seed for full reproducibility
)

curriculum.update_epoch(curriculum_epoch=40)  # Curriculum phase for this batch
```

**Every call is reproducible**:
- Same seed → same cell selection order
- Same epoch → same augmentation probabilities
- Same PNG files will have identical pixels across runs

### Provenance Tracking

The `source_paths` field in `labels.csv` records **exactly which NIH images** contributed to each synthetic sample. This enables:
- **Debugging**: Regenerate a specific synthetic image and verify augmentation
- **Validation**: Track which NIH images appear in training set
- **Fairness**: Ensure balanced representation from Parasitized/Uninfected pools

---

## Why This Approach Works

| Challenge | Solution | Justification |
|---|---|---|
| **Depth occlusion** | Stacking with alpha blending | Matches real thick-smear physics |
| **Partial parasite coverage** | CutMix with soft masks | Realistic high-density field preparation |
| **Optical degradation** | Chromatic aberration, blur, vignetting | Replicate actual foldscope hardware |
| **Stain inconsistency** | HSV hue/saturation jitter | Simulate field-preparation variability |
| **Training instability** | Curriculum scheduling | Gradual increase in task difficulty |
| **Soft labels** | Weighted label blending | Encode uncertainty and mixing ratios |
| **Reproducibility** | Fixed seeds + source tracking | Enable debugging and validation |

---

## Usage

### Generating Synthetic Data

```bash
python -m utils.synthetic_field_ready_dataset \
  --dataset-root nih_data \
  --output-dir synthetic_field_ready \
  --num-samples 10000 \
  --seed 42 \
  --epoch 40
```

### Loading in Training Pipeline

```python
from utils.synthetic_field_ready_dataset import generate_synthetic_dataset

output_dir = generate_synthetic_dataset(
    dataset_root="nih_data",
    output_dir="synthetic_field_ready",
    num_samples=10000,
    seed=42,
    curriculum_epoch=40
)

# Now use synthetic_field_ready/labels.csv and PNG files for training
```

---

## Validation & Metrics

To verify the synthetic dataset quality:

1. **Visual inspection**: Spot-check 10-20 samples from each mode
2. **Label distribution**: Verify class balance (should be ~45-50% infected after mixing)
3. **Augmentation strength**: Inspect curricuum-scheduled samples at epochs 1, 20, 40
4. **Source diversity**: Ensure each source cell is used ~0-2 times (no favorites due to seeding)

---

## Future Enhancements

1. **Adaptive augmentation**: Use validation performance to auto-tune augmentation strengths
2. **Semantic augmentation**: Apply biological transformations (e.g., parasite morphology variations)
3. **Uncertainty quantification**: Track which augmentations hurt model confidence most
4. **Domain adaptation**: Use synthetic→real discrepancy minimization for transfer learning

---

## References

- **NIH Malaria Dataset**: https://lhncbc.nlm.nih.gov/publication/pub9932
- **CutMix**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)
- **Foldscope Optics**: Switz et al., "Low-cost mobile phone microscopy with a reversed mobile phone camera lens" (PLOS Biology 2014)
- **Curriculum Learning**: Bengio et al., "Curriculum Learning" (ICML 2009)

---

**Document Version**: 1.0  
**Last Updated**: May 14, 2026  
**Maintained By**: M.A.L.L.I. Development Team
