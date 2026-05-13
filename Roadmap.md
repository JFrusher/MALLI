# M.A.L.L.I. Project Roadmap
## Mobile Advanced Lightweight Localization & Imaging
**A Biomedical Engineering Initiative for Accessible Malaria Diagnostics**

---

## 1. Project Vision
To bridge the gap between high-end digital pathology and rural point-of-care settings by combining low-cost physical optics with optimized Edge-AI (MobileNetV3).

---

## 2. The ML Progression Pipeline

### Phase A: The Baseline (Perfect Data) - Establishing the "Theoretical Ceiling"
**Goal:** Train on raw NIH dataset to establish the upper bound for model accuracy.
* **Objective:** Understand if MobileNetV3 can differentiate parasites from artifacts under ideal conditions.
* **Steps:**
    * **Dataset Acquisition:** Download the complete NIH Malaria Dataset (27,558 cell images).
    * **Train-Validation-Test Split:** 70-15-15 split with stratification.
    * **Baseline Architecture:** MobileNetV3-Small with ImageNet pre-training.
    * **Training Configuration:**
        * Adam optimizer (lr=1e-3), batch size=32, epochs=50
        * Early stopping on validation accuracy.
        * Track metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
    * **Deliverable:** Baseline accuracy metrics (target: >95% on clean data).
    * **Note:** This phase serves as the "theoretical ceiling" — real-world performance will degrade from this.

---

### Phase B: Domain Adaptation (The "Foldscope Emulator") - Simulating Field Conditions
**Goal:** Augment training data to mimic low-cost optics and thick smears (bunched cells).
* **Objective:** Close the gap between laboratory conditions and real-world Foldscope/field environments.

#### B1: Optical Degradation Augmentations
* **Chromatic Aberration Simulation:**
    * Shift RGB channels by 2-5 pixels to simulate the purple/blue color fringing from cheap lenses.
    * Implementation: Randomly separate channels and recompose with slight offsets.
* **Vignetting (Circular Field-of-View):**
    * Darken the edges of images to mimic the circular field of view of ball-lens microscopy.
    * Gradual radial darkening from center to edges (alpha=0.3-0.7).
* **Laplacian Blur (Manual Focus Variation):**
    * Simulate difficulty of manual focusing in the field by randomly varying focus.
    * Apply variable-strength Laplacian blur (kernel size: 3-9, random offset: ±5 pixels).
* **Motion Blur:**
    * Simulate hand tremor or vibration during image capture.
    * Random motion blur with kernel size 3-7 and angle 0-360°.
* **Gaussian Noise:**
    * Add sensor noise common in low-end camera modules.
    * Noise levels: σ = 0.01-0.05 (normalize to [0,1] range).

#### B2: Thick Smear Simulation (Cell Bunching)
* **CutMix & MixUp Augmentation:**
    * Overlay multiple cell images to simulate overlapping cells found in thick blood smears.
    * **CutMix:** Replace a random rectangular region of one image with a region from another, preserving both labels via soft-labeling.
    * **MixUp:** Linear interpolation of image pairs (λ~Beta(1.0, 1.0)).
    * Vary mixing ratio to simulate different degrees of cell overlap (10-40% overlap).
* **Cell Density Variation:**
    * Randomly stack 2-4 cell images to create dense regions resembling thick smears.
    * Use weighted blending to create realistic overlap effects.

#### B3: Color & Stain Variation
* **Stain Normalization Reversal:**
    * Instead of normalizing, apply controlled stain variations to mimic field-collection inconsistencies.
    * Shift hue by ±10-20°, adjust saturation (0.8-1.2×), brightness (0.9-1.1×).
* **Sample Preparation Artifacts:**
    * Add dust particles (small Gaussian blobs, 1-3 per image).
    * Simulate air bubbles (circular occlusions, 5-10% of image area).

#### B4: Training Strategy
* **Progressive Augmentation:**
    * Epoch 1-20: Baseline NIH data + mild augmentations.
    * Epoch 21-40: Gradual ramp-up of optical distortions and cell mixing.
    * Epoch 41-50: Full augmentation strength.
* **Validation:** Validate on a separate "field-augmented" test set to measure robustness.
* **Deliverable:** Domain-adapted model with >85% accuracy on augmented data (graceful degradation from Phase A).

---

### Phase C: Edge Optimization (The "Lightweight" Tier) - Mobile-Ready Inference
**Goal:** Optimize the trained model for sub-100ms inference on mobile NPUs.
* **Objective:** Ensure the app can process images in real-time without draining battery.

#### C1: Knowledge Distillation
* **Teacher-Student Framework:**
    * **Teacher:** Full MobileNetV3-Small trained in Phase B.
    * **Student:** MobileNetV3-Small with 50% fewer channels (reduced width multiplier: 0.5 instead of 1.0).
    * **Training:** Distill knowledge using cross-entropy loss (student output) + KL-divergence (teacher logits).
    * **Temperature:** T=4.0 for soft targets, α=0.7 (cross-entropy weight).
* **Target:** Match teacher accuracy (>85%) while reducing model size by 40-60%.

#### C2: Pruning & Weight Sharing
* **Structured Pruning:**
    * Remove entire convolutional filters with lowest activation magnitudes.
    * Prune by magnitude: remove filters where L2-norm < threshold (iterative pruning, target: 30-40% filter removal).
* **Unstructured Pruning (Optional):**
    * Sparsify weights to zero where magnitude < threshold.
    * Use magnitude pruning + fine-tuning cycles.
* **Deliverable:** Pruned model 50-60% smaller with <2% accuracy loss.

#### C3: INT8 Quantization
* **Post-Training Quantization (PTQ):**
    * Convert 32-bit float weights and activations to 8-bit integers.
    * Calibrate using a representative subset of training data (1000 images).
    * INT8 is the "sweet spot" for mobile NPUs: achieves 100ms/frame inference on Qualcomm Hexagon, Apple Neural Engine, or MediaTek APU.
* **Quantization-Aware Training (Optional):**
    * Fine-tune the quantized model for 5-10 epochs to recover accuracy.
    * Simulate quantization during training to improve robustness.
* **Deliverable:** INT8 model (model size: <5 MB) with <1% accuracy loss.

#### C4: Model Export & Validation
* **Export Formats:**
    * **TensorFlow Lite (.tflite):** For Android + iOS.
    * **ONNX (.onnx):** Platform-agnostic format.
* **On-Device Testing:**
    * Validate inference speed on target hardware (Android phone + iPhone).
    * Log inference time, power consumption, memory usage.
* **Deliverable:** Model suitable for production deployment (<100ms/frame, <5 MB, <85 MB RAM usage).

---

## 3. Hardware & Optical Integration

### Phase 2: Optical Engineering (The Hardware Bridge)
**Goal:** Achieve ~400x total magnification (optical + digital) for < $20.
* **Lens Selection & Characterization:**
    * *Option A:* **Foldscope** (Paper-based microscopy, ~140x magnification).
    * *Option B:* **Ball Lens Retrofit** (3mm sapphire or glass ball lens, ~130x magnification).
    * *Action:* Test both on representative cell images to validate Phase B augmentations match physical reality.
* **Lighting & Contrast:**
    * Design a 3D-printable clip to align the phone camera with the lens.
    * Implement a simple LED backlighting rig with a diffuser (diffuse white light to reduce shadows).
    * Validate illumination uniformity across the field of view.
* **Image Capture Validation:**
    * Collect 500+ real Foldscope images of parasitized and uninfected cells.
    * Compare against Phase B augmented data to refine augmentation parameters if needed.

---

## 4. Mobile Application Development & Deployment

### Phase 3: Mobile Application Development
**Goal:** Build a seamless, offline-first diagnostic interface.

#### 3.1: Mobile App Architecture
* **Platform:** Android (Kotlin) + iOS (Swift) using TensorFlow Lite.
* **Framework:** Native development with cross-platform CI/CD (GitHub Actions).
* **Real-Time Inference:**
    * Use **TFLite Support Library** for camera stream processing.
    * Implement GPU delegation (Android NN API / CoreML) for faster inference.
* **Core Features:**
    * **Camera Module:** Continuous frame capture at 30 FPS.
    * **Auto-Focus Trigger:** Use Laplacian variance or auto-focus API to detect focus quality.
    * **Auto-Capture:** Automatically snap image when in focus + centered.
    * **Inference Pipeline:** Pass cropped ROI to the quantized model, display result in <500ms.
    * **Offline Mode:** Model bundled with app; no cloud connectivity required.
    * **Logging:** Store diagnostic history locally (SQLite).

#### 3.2: User Interface & UX
* **Diagnosis Screen:**
    * Real-time camera preview with focus quality indicator.
    * Overlay showing confidence score (0-100%).
    * Result: "Parasitized" (red) or "Uninfected" (green) with confidence and recommendation.
* **History & Analytics:**
    * Display past diagnoses with timestamps and location tags (GPS optional).
    * Aggregated statistics: % parasitized, trend over time.
* **Accessibility:**
    * Support for multiple languages (English, French, Swahili).
    * Large buttons, high-contrast UI for field use.

#### 3.3: Robustness & Error Handling
* **Input Validation:**
    * Reject images with low focus quality or poor lighting.
    * Provide actionable feedback: "Reposition the lens," "Adjust lighting," etc.
* **Fallback Mechanisms:**
    * If inference fails, allow manual review and re-capture.
* **Model Versioning:**
    * Support in-app model updates (download new .tflite files over WiFi).

---

### Phase 4: Field Testing & Iterative Refinement
**Goal:** Validate app performance in real-world settings.

#### 4.1: Pilot Studies
* **Test Locations:** Partner with clinics in malaria-endemic regions (e.g., Sub-Saharan Africa).
* **Metrics:**
    * Sensitivity (True Positive Rate): Target >90%.
    * Specificity (True Negative Rate): Target >90%.
    * User satisfaction (SUS score, target >70).
    * Device usability: Battery drain, thermal performance, crash rates.
* **Data Collection:** Collect 1000-2000 real-world images for model retraining.

#### 4.2: Feedback Loop
* **Failure Analysis:** Analyze misclassifications to identify blind spots.
* **Model Retraining:** If sensitivity <90%, retrain Phase C model on failure cases.
* **Hardware Adjustments:** Refine optical alignment or lighting based on field results.

---

## 5. Deployment & Scale

### Phase 5: Production Hardening
**Goal:** Prepare for large-scale deployment.

#### 5.1: Regulatory Compliance
* **Medical Device Classification:** Consult local regulatory bodies (FDA, CE marking, etc.).
* **Clinical Validation:** Conduct formal clinical trials if required.
* **Data Privacy:** Implement GDPR/HIPAA compliance for diagnostic logging.

#### 5.2: Backend Infrastructure (Optional)
* **Cloud Sync:** For voluntary cloud submission, implement encrypted HTTPS upload.
* **Aggregated Analytics:** Anonymous epidemiological tracking for disease surveillance.
* **Model Serving:** TensorFlow Serving or TFLite Model Server for A/B testing new models.

#### 5.3: Distribution & Support
* **App Store Release:** Publish to Google Play Store and Apple App Store.
* **Community Training:** Develop training materials for field workers.
* **Bug Tracking & Updates:** Use GitHub Issues for community feedback and bug reports.

---

## 6. Immediate Next Steps
1. **Environment Setup:** Install Python, PyTorch/TF, and download the NIH dataset (if not done).
2. **Phase A (Baseline):** Train MobileNetV3-Small on clean NIH data; record baseline metrics.
3. **Phase B (Augmentation):** Implement chromatic aberration, vignetting, Laplacian blur, and CutMix augmentations.
4. **Phase C (Optimization):** Conduct knowledge distillation, pruning, and INT8 quantization on the Phase B model.
5. **Hardware Order:** Purchase a Foldscope or 3mm ball lenses for optical validation.
6. **Collect Real Data:** Capture 500+ real Foldscope images to validate augmentation parameters.
7. **App Scaffolding:** Set up Android/iOS project structure with TFLite integration.