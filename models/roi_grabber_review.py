from __future__ import annotations

"""Interactive ROI-grabber review utility.

Use this tool to quality-check ROI proposals and model decisions by browsing
sample images from different smear/class folders with overlay boxes.
"""

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Sequence
import argparse
import logging
import traceback
import random
import zipfile
import io
import tempfile
import shutil
import os
import time

import cv2
import numpy as np
import tensorflow as tf

# Suppress TensorFlow initialization warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING from TensorFlow
tf.get_logger().setLevel("ERROR")  # Only show errors from TensorFlow logger

from .inference import load_model_with_weights
from .detection_pipeline import (
    TilingInferenceEngine,
)
from .roi_grabber import (
    SUPPORTED_EXTENSIONS,
    batch_predict_rois,
    extract_roi_proposals,
    RoiProposal,
    infer_label_from_parts,
    infer_smear_type_by_dimensions,
    infer_smear_type_from_parts,
    non_max_suppression,
    preprocess_for_segmentation,
    segment_thick_smear_watershed,
    segment_thin_smear,
)


@dataclass(frozen=True)
class ReviewSample:
    source_type: str
    source_id: str
    group_key: str
    coverage_key: str
    label_name: str
    smear_type: str


@dataclass(frozen=True)
class SensingVariant:
    name: str
    color: tuple[int, int, int]
    blur_ksize: int
    clahe_clip: float
    threshold_offset: float
    open_iterations: int
    median_ksize: int
    distance_ratio: float
    dilate_iterations: int


@dataclass(frozen=True)
class VariantProposal:
    variant: SensingVariant
    proposal: RoiProposal


@dataclass(frozen=True)
class RoiAnalysisLine:
    rank: int
    variant_name: str
    proposal_score: float | None
    classifier_score: float | None
    decision: str
    box: tuple[int, int, int, int]


SENSING_VARIANT_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 140, 0),
    (0, 200, 255),
    (0, 220, 90),
)
THIN_SENSING_VARIANTS: tuple[SensingVariant, ...] = (
    SensingVariant("thin_sensitive", SENSING_VARIANT_COLORS[0], 3, 2.0, -10.0, 1, 3, 0.0, 0),
    SensingVariant("thin_balanced", SENSING_VARIANT_COLORS[1], 5, 2.0, 0.0, 1, 3, 0.0, 0),
    SensingVariant("thin_strict", SENSING_VARIANT_COLORS[2], 7, 2.5, 12.0, 2, 5, 0.0, 0),
)
THICK_SENSING_VARIANTS: tuple[SensingVariant, ...] = (
    SensingVariant("thick_sensitive", SENSING_VARIANT_COLORS[0], 3, 2.0, -12.0, 1, 0, 0.28, 2),
    SensingVariant("thick_balanced", SENSING_VARIANT_COLORS[1], 5, 2.0, 0.0, 2, 0, 0.35, 3),
    SensingVariant("thick_strict", SENSING_VARIANT_COLORS[2], 7, 2.5, 12.0, 2, 0, 0.42, 4),
)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def summarize_exception(exc: Exception, limit: int = 900) -> str:
    message = f"{type(exc).__name__}: {exc}"
    collapsed = " ".join(message.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def load_model_from_keras_archive_weights(model_path: Path, input_size: tuple[int, int]) -> tf.keras.Model:
    extracted_weights: Path | None = None
    with zipfile.ZipFile(model_path, "r") as archive:
        weight_members = [name for name in archive.namelist() if name.endswith(".weights.h5")]
        if not weight_members:
            raise RuntimeError(f"No embedded .weights.h5 file found inside {model_path.name}")

        # Keras only allows name-based fallback loading on legacy .h5 files.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmpf:
            tmpf.write(archive.read(weight_members[0]))
            extracted_weights = Path(tmpf.name)

    try:
        return load_model_with_weights(
            extracted_weights,
            input_shape=(input_size[0], input_size[1], 3),
            compile_model=False,
        )
    finally:
        try:
            if extracted_weights is not None:
                extracted_weights.unlink(missing_ok=True)
        except Exception:
            pass


def offset_variant_proposal(
    variant_proposal: VariantProposal,
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> VariantProposal:
    x1, y1, x2, y2 = variant_proposal.proposal.box
    mapped_box = (
        max(0, min(image_width, x1 + offset_x)),
        max(0, min(image_height, y1 + offset_y)),
        max(0, min(image_width, x2 + offset_x)),
        max(0, min(image_height, y2 + offset_y)),
    )
    mapped_centroid = (
        max(0, min(image_width - 1, variant_proposal.proposal.centroid[0] + offset_x)),
        max(0, min(image_height - 1, variant_proposal.proposal.centroid[1] + offset_y)),
    )
    return VariantProposal(
        variant=variant_proposal.variant,
        proposal=RoiProposal(box=mapped_box, centroid=mapped_centroid, area=variant_proposal.proposal.area),
    )


def read_image_from_zip(archive: zipfile.ZipFile, member_name: str) -> np.ndarray | None:
    with archive.open(member_name, "r") as handle:
        raw = handle.read()
    image_u8 = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_u8, cv2.IMREAD_COLOR)
    return image


def make_nested_source_id(outer_member: str, inner_member: str) -> str:
    return f"{outer_member}::{inner_member}"


def split_nested_source_id(source_id: str) -> tuple[str, str] | None:
    if "::" not in source_id:
        return None
    outer, inner = source_id.split("::", 1)
    return outer, inner


def infer_group_key(parts_lower: Sequence[str], fallback_name: str) -> str:
    smear = infer_smear_type_from_parts(parts_lower) or "unknown"
    parsed_label = infer_label_from_parts(parts_lower)
    label = parsed_label.lower() if parsed_label is not None else "unknown"
    if smear != "unknown" or label != "unknown":
        return f"{smear}_{label}"
    return fallback_name.lower().replace(" ", "_")


def make_coverage_key(label_name: str, smear_type: str) -> str:
    return f"{smear_type}_{label_name.lower()}"


def crawl_folder_samples(dataset_root: Path, thick_min_dim: int) -> list[ReviewSample]:
    samples: list[ReviewSample] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        parts_lower = [part.lower() for part in path.parts]
        label = infer_label_from_parts(parts_lower)
        if label is None:
            continue

        smear = infer_smear_type_from_parts(parts_lower)
        if smear is None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            smear = infer_smear_type_by_dimensions(h, w, thick_min_dim=thick_min_dim)

        group = infer_group_key(parts_lower, fallback_name=path.parent.name)
        samples.append(
            ReviewSample(
                source_type="folder",
                source_id=str(path),
                group_key=group,
                coverage_key=make_coverage_key(label, smear),
                label_name=label,
                smear_type=smear,
            )
        )
    return samples


def crawl_zip_samples(zip_path: Path, thick_min_dim: int) -> list[ReviewSample]:
    samples: list[ReviewSample] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            entry = Path(info.filename)
            ext = entry.suffix.lower()

            if ext in SUPPORTED_EXTENSIONS:
                parts_lower = [part.lower() for part in entry.parts]
                label = infer_label_from_parts(parts_lower)
                if label is None:
                    continue

                smear = infer_smear_type_from_parts(parts_lower)
                if smear is None:
                    image = read_image_from_zip(archive, info.filename)
                    if image is None:
                        continue
                    h, w = image.shape[:2]
                    smear = infer_smear_type_by_dimensions(h, w, thick_min_dim=thick_min_dim)

                group = infer_group_key(parts_lower, fallback_name=entry.parent.name)
                samples.append(
                    ReviewSample(
                        source_type="zip",
                        source_id=info.filename,
                        group_key=group,
                        coverage_key=make_coverage_key(label, smear),
                        label_name=label,
                        smear_type=smear,
                    )
                )
                continue

            if ext != ".zip":
                continue

            # Stream nested zip to memory (BytesIO) to avoid disk space issues.
            # Falls back to disk temp file if memory is insufficient.
            try:
                with archive.open(info.filename) as nested_fp:
                    nested_bytes = io.BytesIO(nested_fp.read())
                with zipfile.ZipFile(nested_bytes, "r") as nested:
                    for nested_info in nested.infolist():
                        if nested_info.is_dir():
                            continue
                        nested_entry = Path(nested_info.filename)
                        if nested_entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
                            continue

                        combined_parts = [part.lower() for part in entry.parts] + [
                            part.lower() for part in nested_entry.parts
                        ]
                        label = infer_label_from_parts(combined_parts)
                        if label is None:
                            continue

                        smear = infer_smear_type_from_parts(combined_parts)
                        if smear is None:
                            image = read_image_from_zip(nested, nested_info.filename)
                            if image is None:
                                continue
                            h, w = image.shape[:2]
                            smear = infer_smear_type_by_dimensions(h, w, thick_min_dim=thick_min_dim)

                        group = infer_group_key(combined_parts, fallback_name=entry.stem)
                        samples.append(
                            ReviewSample(
                                source_type="nested_zip",
                                source_id=make_nested_source_id(info.filename, nested_info.filename),
                                group_key=group,
                                coverage_key=make_coverage_key(label, smear),
                                label_name=label,
                                smear_type=smear,
                            )
                        )
            except MemoryError:
                # Fallback to disk if nested zip is too large for memory
                logging.warning(f"Nested zip too large for memory, using disk temp: {info.filename}")
                with archive.open(info.filename) as nested_fp:
                    tmpf = tempfile.NamedTemporaryFile(delete=False)
                    try:
                        shutil.copyfileobj(nested_fp, tmpf)
                        tmpf.close()
                        with zipfile.ZipFile(tmpf.name, "r") as nested:
                            for nested_info in nested.infolist():
                                if nested_info.is_dir():
                                    continue
                                nested_entry = Path(nested_info.filename)
                                if nested_entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
                                    continue

                                combined_parts = [part.lower() for part in entry.parts] + [
                                    part.lower() for part in nested_entry.parts
                                ]
                                label = infer_label_from_parts(combined_parts)
                                if label is None:
                                    continue

                                smear = infer_smear_type_from_parts(combined_parts)
                                if smear is None:
                                    image = read_image_from_zip(nested, nested_info.filename)
                                    if image is None:
                                        continue
                                    h, w = image.shape[:2]
                                    smear = infer_smear_type_by_dimensions(h, w, thick_min_dim=thick_min_dim)

                                group = infer_group_key(combined_parts, fallback_name=entry.stem)
                                samples.append(
                                    ReviewSample(
                                        source_type="nested_zip",
                                        source_id=make_nested_source_id(info.filename, nested_info.filename),
                                        group_key=group,
                                        coverage_key=make_coverage_key(label, smear),
                                        label_name=label,
                                        smear_type=smear,
                                    )
                                )
                    finally:
                        try:
                            os.remove(tmpf.name)
                        except Exception:
                            pass
    return samples


def stratified_sample_by_group(samples: Sequence[ReviewSample], per_group: int, seed: int) -> list[ReviewSample]:
    groups: dict[str, list[ReviewSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_key, []).append(sample)

    rng = random.Random(seed)
    selected: list[ReviewSample] = []
    for key in sorted(groups.keys()):
        pool = groups[key]
        if not pool:
            continue
        k = min(per_group, len(pool))
        selected.extend(rng.sample(pool, k))

    rng.shuffle(selected)
    return selected


def stratified_sample_by_coverage(samples: Sequence[ReviewSample], per_group: int, seed: int) -> list[ReviewSample]:
    coverage_groups: dict[str, list[ReviewSample]] = {}
    for sample in samples:
        coverage_groups.setdefault(sample.coverage_key, []).append(sample)

    rng = random.Random(seed)
    selected: list[ReviewSample] = []
    for key in sorted(coverage_groups.keys()):
        pool = coverage_groups[key]
        if not pool:
            continue
        k = min(per_group, len(pool))
        selected.extend(rng.sample(pool, k))

    rng.shuffle(selected)
    return selected


def load_sample_image(
    sample: ReviewSample,
    zip_path: Path | None,
    zip_cache: zipfile.ZipFile | None,
    nested_zip_cache: dict[str, zipfile.ZipFile] | None,
) -> np.ndarray | None:
    if sample.source_type == "folder":
        return cv2.imread(sample.source_id, cv2.IMREAD_COLOR)

    if zip_path is None:
        return None

    if sample.source_type == "nested_zip":
        parts = split_nested_source_id(sample.source_id)
        if parts is None:
            return None
        outer_member, inner_member = parts

        if nested_zip_cache is not None and outer_member in nested_zip_cache:
            return read_image_from_zip(nested_zip_cache[outer_member], inner_member)

        if zip_cache is not None:
            nested_bytes = zip_cache.read(outer_member)
        else:
            with zipfile.ZipFile(zip_path, "r") as archive:
                nested_bytes = archive.read(outer_member)

        with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested:
            return read_image_from_zip(nested, inner_member)

    if zip_cache is None:
        with zipfile.ZipFile(zip_path, "r") as archive:
            return read_image_from_zip(archive, sample.source_id)

    return read_image_from_zip(zip_cache, sample.source_id)


def normalize_kernel_size(size: int) -> int:
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


def preprocess_for_sensing(image_bgr: np.ndarray, clahe_clip: float, blur_ksize: int) -> np.ndarray:
    green = image_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    normalized = clahe.apply(green)
    blur_ksize = normalize_kernel_size(blur_ksize)
    if blur_ksize > 1:
        return cv2.GaussianBlur(normalized, (blur_ksize, blur_ksize), sigmaX=0)
    return normalized


def threshold_foreground(preprocessed: np.ndarray, threshold_offset: float) -> np.ndarray:
    otsu_threshold, _ = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = float(np.clip(otsu_threshold + threshold_offset, 0.0, 255.0))
    _, thresholded = cv2.threshold(preprocessed, cutoff, 255, cv2.THRESH_BINARY_INV)
    return thresholded


def segment_thin_smear_variant(image_bgr: np.ndarray, variant: SensingVariant) -> np.ndarray:
    preprocessed = preprocess_for_sensing(image_bgr, clahe_clip=variant.clahe_clip, blur_ksize=variant.blur_ksize)
    thresholded = threshold_foreground(preprocessed, threshold_offset=variant.threshold_offset)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=max(1, variant.open_iterations))
    if variant.median_ksize > 1:
        opened = cv2.medianBlur(opened, normalize_kernel_size(variant.median_ksize))
    return opened


def segment_thick_smear_variant(image_bgr: np.ndarray, variant: SensingVariant) -> np.ndarray:
    preprocessed = preprocess_for_sensing(image_bgr, clahe_clip=variant.clahe_clip, blur_ksize=variant.blur_ksize)
    thresholded = threshold_foreground(preprocessed, threshold_offset=variant.threshold_offset)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=max(1, variant.open_iterations))
    sure_bg = cv2.dilate(opening, kernel, iterations=max(1, variant.dilate_iterations))

    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0.0:
        return opening

    distance_ratio = float(np.clip(variant.distance_ratio, 0.1, 0.7))
    _, sure_fg = cv2.threshold(distance, distance_ratio * max_distance, 255, 0)
    sure_fg_u8 = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg_u8.astype(np.uint8))  # type: ignore[arg-type]

    _, markers = cv2.connectedComponents(sure_fg_u8.astype(np.uint8))  # type: ignore[arg-type]
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_markers = cv2.watershed(image_bgr.copy(), markers)
    mask = np.zeros(preprocessed.shape, dtype=np.uint8)
    mask[watershed_markers > 1] = 255
    return mask


def sensing_variants_for_smear(smear_type: str) -> tuple[SensingVariant, ...]:
    return THICK_SENSING_VARIANTS if smear_type == "thick" else THIN_SENSING_VARIANTS


def extract_variant_proposals(
    image: np.ndarray,
    variant: SensingVariant,
    smear_type: str,
    roi_size: int,
    min_blob_area: float,
    max_blob_area: float,
) -> list[VariantProposal]:
    if smear_type == "thick":
        mask = segment_thick_smear_variant(image, variant)
    else:
        mask = segment_thin_smear_variant(image, variant)

    proposals = extract_roi_proposals(
        mask,
            image_shape=(int(image.shape[0]), int(image.shape[1]), int(image.shape[2])),
        roi_size=roi_size,
        min_blob_area=min_blob_area,
        max_blob_area=max_blob_area,
    )
    return [VariantProposal(variant=variant, proposal=proposal) for proposal in proposals]


def draw_variant_overlay(
    image_bgr: np.ndarray,
    proposals: Sequence[VariantProposal],
    classifier_scores: np.ndarray | None,
    proposal_scores: np.ndarray | None,
    kept_indices: set[int],
    positive_indices: set[int],
    display_threshold: float | None = None,
) -> np.ndarray:
    overlay = image_bgr.copy()

    for idx, variant_proposal in enumerate(proposals):
        x1, y1, x2, y2 = variant_proposal.proposal.box
        proposal_score = None
        if proposal_scores is not None and proposal_scores.size > idx:
            proposal_score = float(proposal_scores[idx])

        classifier_score = None
        if classifier_scores is not None and classifier_scores.size > idx:
            classifier_score = float(classifier_scores[idx])

        label_parts = [variant_proposal.variant.name]
        if proposal_score is not None:
            label_parts.append(f"src={proposal_score:.2f}")
        if classifier_score is not None:
            label_parts.append(f"mnet={classifier_score:.2f}")
            label_parts.append("INF" if classifier_score >= (display_threshold if display_threshold is not None else 0.3) else "HLT")

        label = " | ".join(label_parts)
        confidence_value = 0.0
        if classifier_score is not None:
            confidence_value = classifier_score
        elif proposal_score is not None:
            confidence_value = proposal_score

        color = confidence_to_bgr(confidence_value)
        if idx in positive_indices:
            thickness = 2 + int(round(confidence_value * 3))
        elif idx in kept_indices:
            thickness = 1 + int(round(confidence_value * 2))
        else:
            thickness = 1

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_bg_x2 = min(overlay.shape[1] - 1, x1 + text_size[0] + 6)
        text_bg_y1 = max(0, y1 - text_size[1] - 10)
        cv2.rectangle(overlay, (x1, text_bg_y1), (text_bg_x2, max(0, y1 - 2)), color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            overlay,
            label,
            (x1 + 3, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


def build_variant_legend(
    variants: Sequence[SensingVariant],
    raw_counts: Counter[str],
    kept_counts: Counter[str],
    positive_counts: Counter[str],
    model_enabled: bool,
) -> list[str]:
    legend_lines: list[str] = []
    for variant in variants:
        raw = raw_counts.get(variant.name, 0)
        kept = kept_counts.get(variant.name, 0)
        if model_enabled:
            positive = positive_counts.get(variant.name, 0)
            legend_lines.append(f"{variant.name}: raw={raw} kept={kept} pos={positive}")
        else:
            legend_lines.append(f"{variant.name}: raw={raw} kept={kept}")
    return legend_lines


def build_roi_analysis_lines(
    proposals: Sequence[VariantProposal],
    proposal_scores: np.ndarray | None,
    classifier_scores: np.ndarray | None,
    kept_indices: Sequence[int],
    threshold: float,
    max_lines: int,
) -> list[str]:
    if not proposals:
        return []

    order = list(kept_indices)
    if classifier_scores is not None:
        order.sort(key=lambda idx: float(classifier_scores[idx]), reverse=True)
    elif proposal_scores is not None:
        order.sort(key=lambda idx: float(proposal_scores[idx]), reverse=True)

    lines: list[str] = []
    for rank, idx in enumerate(order[:max_lines], start=1):
        proposal = proposals[idx]
        proposal_score = float(proposal_scores[idx]) if proposal_scores is not None and proposal_scores.size > idx else None
        classifier_score = float(classifier_scores[idx]) if classifier_scores is not None and classifier_scores.size > idx else None
        decision = "N/A"
        if classifier_score is not None:
            decision = "INF" if classifier_score >= threshold else "HLT"
        x1, y1, x2, y2 = proposal.proposal.box
        score_bits: list[str] = []
        if proposal_score is not None:
            score_bits.append(f"src={proposal_score:.2f}")
        if classifier_score is not None:
            score_bits.append(f"mnet={classifier_score:.2f}")
        score_text = " ".join(score_bits) if score_bits else "no-scores"
        lines.append(
            f"{rank:02d} {proposal.variant.name} {decision} {score_text} box=({x1},{y1},{x2},{y2})"
        )
    return lines


def confidence_to_bgr(confidence: float) -> tuple[int, int, int]:
    """Map confidence to a vivid heatmap color."""
    value = int(round(float(np.clip(confidence, 0.0, 1.0)) * 255.0))
    color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def compute_overlay_sahi_pipeline(
    image: np.ndarray,
    smear_type: str,
    model: tf.keras.Model | None,
    threshold: float,
    tile_size: int,
    overlap_ratio: float,
    roi_size: int,
    min_blob_area: float,
    max_blob_area: float,
    nms_threshold: float,
    model_input_size: tuple[int, int],
    batch_size: int,
    max_roi_analysis_lines: int,
) -> tuple[np.ndarray, int, int, float, list[str], list[str]]:
    """Compute overlay using a tiled variant of the existing sensing pipeline.

    SAHI is most useful here as a proposal-expansion layer: tile the image, run the
    same sensing variants on each tile, merge proposals across tile boundaries,
    then classify the deduplicated ROIs once on the full image.
    """
    tiler = TilingInferenceEngine(tile_size=tile_size, overlap_ratio=overlap_ratio)

    # Build tile-local sensing proposals using the same segmentation logic as the
    # baseline pipeline, then map them back into full-image coordinates.
    # Reflect padding reduces tile-edge artifacts; explicit edge rejection keeps
    # duplicate fragments from winning over centered, complete cells.
    variants = sensing_variants_for_smear(smear_type)
    all_variant_proposals: list[VariantProposal] = []
    proposal_quality_scores: list[float] = []
    raw_counts: Counter[str] = Counter()
    edge_margin = max(roi_size // 2, int(tile_size * overlap_ratio * 0.35))
    pad = edge_margin

    for tile_info in tiler.generate_tiles(image.shape[0], image.shape[1]):
        tile_x = tile_info["x"]
        tile_y = tile_info["y"]
        tile_w = tile_info["width"]
        tile_h = tile_info["height"]
        tile_crop = image[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
        if tile_crop.size == 0:
            continue
        tile = cv2.copyMakeBorder(
            tile_crop,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_REFLECT_101,
        )

        for variant in variants:
            proposals = extract_variant_proposals(
                image=tile,
                variant=variant,
                smear_type=smear_type,
                roi_size=roi_size,
                min_blob_area=min_blob_area,
                max_blob_area=max_blob_area,
            )
            raw_counts[variant.name] += len(proposals)
            for proposal in proposals:
                local_cx = proposal.proposal.centroid[0] - pad
                local_cy = proposal.proposal.centroid[1] - pad

                # Keep proposals that are sufficiently far from internal tile edges.
                # Edge-touching proposals are usually fragments that overlap with a
                # better centered copy in a neighboring tile.
                if tile_x > 0 and local_cx < edge_margin:
                    continue
                if tile_y > 0 and local_cy < edge_margin:
                    continue
                if tile_x + tile_w < image.shape[1] and local_cx > (tile_w - edge_margin):
                    continue
                if tile_y + tile_h < image.shape[0] and local_cy > (tile_h - edge_margin):
                    continue

                mapped = offset_variant_proposal(
                    proposal,
                    tile_x - pad,
                    tile_y - pad,
                    image.shape[1],
                    image.shape[0],
                )
                edge_distance = min(
                    max(0.0, local_cx),
                    max(0.0, local_cy),
                    max(0.0, tile_w - local_cx),
                    max(0.0, tile_h - local_cy),
                )
                center_bias = 1.0 + (edge_distance / max(1.0, float(edge_margin)))
                all_variant_proposals.append(mapped)
                proposal_quality_scores.append(float(proposal.proposal.area) * center_bias)

    if not all_variant_proposals:
        return image.copy(), 0, 0, 0.0, build_variant_legend(variants, raw_counts, Counter(), Counter(), model is not None), []

    # Post-processing across tile boundaries before classification keeps the model
    # from seeing duplicate crops multiple times.
    boxes = np.array([proposal.proposal.box for proposal in all_variant_proposals], dtype=np.float32)
    quality_scores = np.array(proposal_quality_scores, dtype=np.float32)
    kept_indices_list = non_max_suppression(boxes, quality_scores, iou_threshold=nms_threshold)
    kept_variant_proposals = [all_variant_proposals[idx] for idx in kept_indices_list]

    if model is None:
        probabilities = None
    else:
        probabilities = batch_predict_rois(
            model=model,
            image_bgr=image,
            proposals=[proposal.proposal for proposal in kept_variant_proposals],
            model_input_size=model_input_size,
            batch_size=batch_size,
        )

    kept_indices = set(range(len(kept_variant_proposals)))
    positive_indices: set[int] = set()
    if probabilities is not None:
        positive_indices = {idx for idx in kept_indices if float(probabilities[idx]) >= threshold}

    kept_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for idx in kept_indices:
        variant_name = kept_variant_proposals[idx].variant.name
        kept_counts[variant_name] += 1
    for idx in positive_indices:
        variant_name = kept_variant_proposals[idx].variant.name
        positive_counts[variant_name] += 1

    overlay = draw_variant_overlay(
        image_bgr=image,
        proposals=kept_variant_proposals,
        classifier_scores=probabilities,
        proposal_scores=None,
        kept_indices=kept_indices,
        positive_indices=positive_indices,
        display_threshold=threshold,
    )

    total_cells = len(kept_indices)
    parasites = len(positive_indices)
    parasitemia = (100.0 * parasites / total_cells) if total_cells > 0 else 0.0
    legend_lines = build_variant_legend(variants, raw_counts, kept_counts, positive_counts, model is not None)
    analysis_lines = build_roi_analysis_lines(
        proposals=kept_variant_proposals,
        proposal_scores=None,
        classifier_scores=probabilities,
        kept_indices=list(range(len(kept_variant_proposals))),
        threshold=threshold,
        max_lines=max_roi_analysis_lines,
    )
    return overlay, total_cells, parasites, parasitemia, legend_lines, analysis_lines


def compute_overlay(
    sample: ReviewSample,
    image: np.ndarray,
    model: tf.keras.Model | None,
    threshold: float,
    roi_size: int,
    model_input_size: tuple[int, int],
    batch_size: int,
    nms_iou_threshold: float,
    min_blob_area: float,
    max_blob_area: float,
    use_yolo: bool = False,
    yolo_weights: str | None = None,
    yolo_conf: float = 0.3,
    yolo_nms_iou: float = 0.5,
    max_roi_analysis_lines: int = 10,
    use_sahi: bool = False,
    sahi_tile_size: int = 640,
    sahi_overlap: float = 0.2,
    sahi_nms_threshold: float = 0.5,
) -> tuple[np.ndarray, int, int, float, list[str], list[str]]:
    # Use SAHI pipeline if requested
    if use_sahi:
        return compute_overlay_sahi_pipeline(
            image=image,
            smear_type=sample.smear_type,
            model=model,
            threshold=threshold,
            tile_size=sahi_tile_size,
            overlap_ratio=sahi_overlap,
            roi_size=roi_size,
            min_blob_area=min_blob_area,
            max_blob_area=max_blob_area,
            nms_threshold=sahi_nms_threshold,
            model_input_size=model_input_size,
            batch_size=batch_size,
            max_roi_analysis_lines=max_roi_analysis_lines,
        )
    variant_proposals: list[VariantProposal] = []
    raw_counts: Counter[str] = Counter()
    proposal_scores: np.ndarray | None = None

    if use_yolo:
        try:
            from .roi_grabber_yolo import yolov8_detect_detections
        except Exception:
            yolov8_detect_detections = None

        if yolov8_detect_detections is None or yolo_weights is None:
            # Fallback to original sensing variants
            variants = sensing_variants_for_smear(sample.smear_type)
            for variant in variants:
                proposals = extract_variant_proposals(
                    image=image,
                    variant=variant,
                    smear_type=sample.smear_type,
                    roi_size=roi_size,
                    min_blob_area=min_blob_area,
                    max_blob_area=max_blob_area,
                )
                raw_counts[variant.name] = len(proposals)
                variant_proposals.extend(proposals)
        else:
            # Use YOLO detections and represent them as a single "yolo" variant.
            # If the YOLO detection call fails at runtime (missing ultralytics, etc.)
            # gracefully fall back to sensing variants instead of raising.
            try:
                yolo_detections = yolov8_detect_detections(
                    image_bgr=image,
                    yolo_weights=yolo_weights,
                    conf_thresh=yolo_conf,
                    nms_iou=yolo_nms_iou,
                    roi_size=roi_size,
                )
            except Exception as exc:
                logging.warning(
                    "YOLO detection failed at runtime (%s). Falling back to sensing variants.",
                    exc,
                )
                yolo_detections = None

            if not yolo_detections:
                # Fallback to original sensing variants when YOLO is unavailable or failed
                variants = sensing_variants_for_smear(sample.smear_type)
                for variant in variants:
                    proposals = extract_variant_proposals(
                        image=image,
                        variant=variant,
                        smear_type=sample.smear_type,
                        roi_size=roi_size,
                        min_blob_area=min_blob_area,
                        max_blob_area=max_blob_area,
                    )
                    raw_counts[variant.name] = len(proposals)
                    variant_proposals.extend(proposals)
                proposal_scores = None
            else:
                yolo_variant = SensingVariant("yolo", (0, 0, 255), 0, 0.0, 0.0, 0, 0, 0.0, 0)
                raw_counts[yolo_variant.name] = len(yolo_detections)
                variant_proposals = [
                    VariantProposal(
                        variant=yolo_variant,
                        proposal=RoiProposal(box=det.box, centroid=det.centroid, area=det.area),
                    )
                    for det in yolo_detections
                ]
                proposal_scores = np.array([det.confidence for det in yolo_detections], dtype=np.float32)
                variants = (yolo_variant,)
    else:
        variants = sensing_variants_for_smear(sample.smear_type)
        for variant in variants:
            proposals = extract_variant_proposals(
                image=image,
                variant=variant,
                smear_type=sample.smear_type,
                roi_size=roi_size,
                min_blob_area=min_blob_area,
                max_blob_area=max_blob_area,
            )
            raw_counts[variant.name] = len(proposals)
            variant_proposals.extend(proposals)

    if not variant_proposals:
        return (
            image.copy(),
            0,
            0,
            0.0,
            build_variant_legend(variants, raw_counts, Counter(), Counter(), model is not None),
            [],
        )

    boxes = np.array([proposal.proposal.box for proposal in variant_proposals], dtype=np.float32)
    if model is None:
        scores = np.array([proposal.proposal.area for proposal in variant_proposals], dtype=np.float32)
        probabilities: np.ndarray | None = None
    else:
        probabilities = batch_predict_rois(
            model=model,
            image_bgr=image,
            proposals=[proposal.proposal for proposal in variant_proposals],
            model_input_size=model_input_size,
            batch_size=batch_size,
        )
        scores = probabilities

    kept_indices_list = non_max_suppression(boxes, scores, iou_threshold=nms_iou_threshold)
    kept_indices = set(kept_indices_list)

    positive_indices: set[int] = set()
    if probabilities is not None:
        positive_indices = {idx for idx in kept_indices_list if float(probabilities[idx]) >= threshold}

    kept_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for idx in kept_indices_list:
        variant_name = variant_proposals[idx].variant.name
        kept_counts[variant_name] += 1
    for idx in positive_indices:
        variant_name = variant_proposals[idx].variant.name
        positive_counts[variant_name] += 1

    overlay = draw_variant_overlay(
        image_bgr=image,
        proposals=variant_proposals,
        classifier_scores=probabilities,
        proposal_scores=proposal_scores,
        kept_indices=kept_indices,
        positive_indices=positive_indices,
        # Use the classifier threshold for visualizing INF/HLT and coloring
        display_threshold=threshold,
    )

    total_cells = len(kept_indices_list)
    parasites = len(positive_indices)
    parasitemia = (100.0 * parasites / total_cells) if total_cells > 0 else 0.0
    legend_lines = build_variant_legend(variants, raw_counts, kept_counts, positive_counts, model is not None)
    analysis_lines = build_roi_analysis_lines(
        proposals=variant_proposals,
        proposal_scores=proposal_scores,
        classifier_scores=probabilities,
        kept_indices=kept_indices_list,
        threshold=threshold,
        max_lines=max_roi_analysis_lines,
    )
    return overlay, total_cells, parasites, parasitemia, legend_lines, analysis_lines


def annotate_hud(
    overlay: np.ndarray,
    sample: ReviewSample,
    index: int,
    total: int,
    total_cells: int,
    parasites: int,
    parasitemia: float,
    threshold: float,
    active_group: str,
    variant_legend: Sequence[str],
    roi_analysis_lines: Sequence[str],
) -> np.ndarray:
    canvas = overlay.copy()
    header_lines = [
        f"Image {index + 1}/{total} | group={sample.group_key} | active_group={active_group}",
        f"label={sample.label_name} | smear={sample.smear_type} | cells={total_cells} | parasites={parasites} | parasitemia={parasitemia:.2f}%",
        f"threshold={threshold:.2f} | keys: n/p next-prev, g cycle group, +/- threshold, s save, q quit",
    ]
    header_lines.extend(variant_legend)
    if roi_analysis_lines:
        header_lines.append("ROI analysis (top MobileNetV3-positive ROIs):")
        header_lines.extend(roi_analysis_lines)

    y = 20
    for line in header_lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive review for ROI grabber overlays")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to extracted dataset root")
    parser.add_argument("--zip-path", type=Path, default=Path("dataverse_files.zip"), help="Path to Dataverse ZIP")
    parser.add_argument("--use-zip", action="store_true", help="Force reading from ZIP")
    parser.add_argument("--per-group", type=int, default=30, help="Number of sampled images per folder-group")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/last_mobilenetv3_small.keras"),
        help="MobileNetV3 weights or saved model path",
    )
    parser.add_argument("--disable-model", action="store_true", help="Show ROI boxes only, without classifier labels")
    parser.add_argument("--threshold", type=float, default=0.85, help="Infected threshold")
    parser.add_argument("--roi-size", type=int, default=128, help="Minimum ROI side length for extracted crops")
    parser.add_argument("--model-input-size", type=int, default=224, help="Model input size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ROI inference")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.30, help="NMS IoU threshold")
    parser.add_argument("--thick-min-dim", type=int, default=512, help="Fallback thick-smear min dimension")
    parser.add_argument("--min-blob-area", type=float, default=80.0, help="Minimum blob area")
    parser.add_argument("--max-blob-area", type=float, default=50000.0, help="Maximum blob area")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/roi_review"), help="Directory for saved overlays")
    parser.add_argument("--window-name", type=str, default="MALLI ROI Reviewer", help="OpenCV window name")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode (save images, no GUI)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode (requires display)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    parser.add_argument("--use-yolo", action="store_true", help="Use YOLOv8 proposals instead of sensing variants")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLO weights path (default yolov8n.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.3, help="YOLO detection confidence threshold")
    parser.add_argument("--yolo-nms-iou", type=float, default=0.5, help="YOLO NMS IoU threshold for proposals")
    parser.add_argument("--roboflow-api-key", type=str, default=None, help="Roboflow API key to download a trained model")
    parser.add_argument("--roboflow-workspace", type=str, default=None, help="Roboflow workspace name")
    parser.add_argument("--roboflow-project", type=str, default=None, help="Roboflow project name")
    parser.add_argument("--roboflow-version", type=int, default=1, help="Roboflow project version to download")
    parser.add_argument("--roboflow-download-dir", type=str, default=None, help="Local directory to download Roboflow export")
    parser.add_argument("--max-roi-analysis-lines", type=int, default=10, help="Maximum ROI analysis lines shown in the HUD")
    # SAHI Pipeline arguments (new Dart-compatible pipeline)
    parser.add_argument(
        "--use-sahi-pipeline",
        action="store_true",
        help="Use tiled sensing pipeline (overlapping tiles + baseline ROI proposals + cross-tile NMS)",
    )
    parser.add_argument("--sahi-tile-size", type=int, default=640, help="SAHI tile size in pixels")
    parser.add_argument("--sahi-overlap", type=float, default=0.2, help="SAHI tile overlap ratio (0.0-1.0)")
    parser.add_argument("--sahi-nms-threshold", type=float, default=0.5, help="SAHI NMS IoU threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    
    # Clean up stale temp files from previous runs to free disk space
    temp_dir = Path(tempfile.gettempdir())
    try:
        import glob
        stale_temp_files = glob.glob(str(temp_dir / "tmp*"))
        cleaned_count = 0
        for temp_file in stale_temp_files[:100]:  # Limit to avoid excessive I/O
            try:
                if os.path.isfile(temp_file):
                    file_age_hours = (time.time() - os.path.getmtime(temp_file)) / 3600
                    if file_age_hours > 1:  # Only remove files older than 1 hour
                        os.remove(temp_file)
                        cleaned_count += 1
            except Exception:
                pass
        if cleaned_count > 0:
            logging.info(f"Cleaned up {cleaned_count} stale temp files to free disk space")
    except Exception as e:
        logging.debug(f"Temp cleanup skipped: {e}")

    def load_review_model(model_path: Path, input_size: tuple[int, int]) -> tf.keras.Model:
        logging.info(f"Model file exists: {model_path.exists()}, size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes")
        logging.debug(
            "Model load context: path=%s suffix=%s input_size=%s tensorflow=%s",
            model_path,
            model_path.suffix,
            input_size,
            tf.__version__,
        )
        
        if model_path.suffix.lower() == ".weights.h5" or model_path.name.endswith(".weights.h5"):
            logging.info("Loading model as .weights.h5 format")
            return load_model_with_weights(
                model_path,
                input_shape=(input_size[0], input_size[1], 3),
                compile_model=False,
            )

        logging.info(f"Loading model as {model_path.suffix} format using tf.keras.models.load_model()")
        try:
            return tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
        except TypeError as e:
            logging.debug(f"First load attempt failed with TypeError (safe_mode not supported): {e}")
            try:
                return tf.keras.models.load_model(str(model_path), compile=False)
            except Exception as fallback_exc:
                logging.warning(
                    "Direct model load failed for %s after safe_mode fallback: %s",
                    model_path.name,
                    summarize_exception(fallback_exc),
                )
                try:
                    import tf_keras as legacy_keras
                except ImportError:
                    legacy_keras = None

                if legacy_keras is not None:
                    try:
                        logging.info("Trying legacy tf_keras loader for %s", model_path.name)
                        return legacy_keras.models.load_model(str(model_path), compile=False)
                    except Exception as legacy_exc:
                        logging.warning(
                            "Legacy tf_keras load failed for %s: %s",
                            model_path.name,
                            summarize_exception(legacy_exc),
                        )
                if zipfile.is_zipfile(model_path):
                    logging.info("Attempting rebuild-from-weights fallback for archive model %s", model_path.name)
                    return load_model_from_keras_archive_weights(model_path, input_size)
                raise
        except Exception as e:
            logging.warning(
                "Direct model load failed for %s: %s",
                model_path.name,
                summarize_exception(e),
            )
            try:
                import tf_keras as legacy_keras
            except ImportError:
                legacy_keras = None

            if legacy_keras is not None:
                try:
                    logging.info("Trying legacy tf_keras loader for %s", model_path.name)
                    return legacy_keras.models.load_model(str(model_path), compile=False)
                except Exception as legacy_exc:
                    logging.warning(
                        "Legacy tf_keras load failed for %s: %s",
                        model_path.name,
                        summarize_exception(legacy_exc),
                    )
            if zipfile.is_zipfile(model_path):
                logging.info("Attempting rebuild-from-weights fallback for archive model %s", model_path.name)
                try:
                    return load_model_from_keras_archive_weights(model_path, input_size)
                except Exception as fallback_exc:
                    logging.error(
                        "Archive fallback failed for %s: %s",
                        model_path.name,
                        summarize_exception(fallback_exc),
                    )
                    if args.verbose:
                        logging.debug(
                            "Full model-load traceback for %s:\n%s",
                            model_path.name,
                            "".join(traceback.format_exception(type(fallback_exc), fallback_exc, fallback_exc.__traceback__)),
                        )
                    raise RuntimeError(
                        f"Failed to load model archive {model_path.name}. See verbose logs for the full traceback."
                    ) from fallback_exc
            if args.verbose:
                logging.debug(
                    "Full model-load traceback for %s:\n%s",
                    model_path.name,
                    "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                )
            raise RuntimeError(
                f"Failed to load model {model_path.name}: {summarize_exception(e)}"
            ) from e

    source_samples: list[ReviewSample]
    use_zip = bool(args.use_zip)
    if not use_zip and args.dataset_root is not None and args.dataset_root.exists():
        source_samples = crawl_folder_samples(args.dataset_root, thick_min_dim=args.thick_min_dim)
    else:
        if not args.zip_path.exists():
            raise FileNotFoundError(
                f"ZIP path not found: {args.zip_path}. Provide --dataset-root or a valid --zip-path."
            )
        
        # Check available disk space before processing large zips
        zip_parent = args.zip_path.parent
        disk_stats = shutil.disk_usage(str(zip_parent))
        free_gb = disk_stats.free / (1024**3)
        logging.info(f"Disk space available: {free_gb:.2f} GB on {zip_parent}")
        if free_gb < 0.5:
            logging.warning(
                f"Warning: Only {free_gb:.2f} GB free on disk. Processing may fail during zip extraction. "
                "Consider freeing space or moving to a larger disk."
            )
        
        source_samples = crawl_zip_samples(args.zip_path, thick_min_dim=args.thick_min_dim)
        use_zip = True

    if not source_samples:
        raise RuntimeError("No review samples were discovered.")

    sampled = stratified_sample_by_coverage(source_samples, per_group=args.per_group, seed=args.seed)
    if not sampled:
        raise RuntimeError("No sampled items were selected. Increase --per-group.")

    model: tf.keras.Model | None = None
    if not args.disable_model:
        logging.info("Loading model: %s", args.model_path)
        if not args.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path.absolute()}")
        try:
            model = load_review_model(args.model_path, (args.model_input_size, args.model_input_size))
        except Exception as exc:
            logging.error(
                "Model loading failed: %s. Re-run with --verbose for the full traceback.",
                summarize_exception(exc),
            )
            
            # Fallback: try alternative model format
            fallback_path = None
            if args.model_path.suffix == ".keras":
                fallback_path = args.model_path.with_suffix(".weights.h5")
            elif args.model_path.suffix == ".h5" and "weights" not in args.model_path.name:
                fallback_path = Path(str(args.model_path).replace(".h5", ".weights.h5"))
            
            if fallback_path and fallback_path.exists():
                logging.warning(f"Trying fallback model: {fallback_path}")
                try:
                    model = load_review_model(fallback_path, (args.model_input_size, args.model_input_size))
                    logging.info(f"Successfully loaded fallback model: {fallback_path}")
                except Exception as exc2:
                    logging.error(
                        "Fallback model also failed: %s",
                        summarize_exception(exc2),
                    )
                    if args.verbose:
                        logging.debug(
                            "Full fallback traceback for %s:\n%s",
                            fallback_path,
                            "".join(traceback.format_exception(type(exc2), exc2, exc2.__traceback__)),
                        )
                    raise RuntimeError(
                        f"Failed to load model (primary: {args.model_path}, fallback: {fallback_path}). "
                        f"Error: {summarize_exception(exc)}"
                    ) from exc
            else:
                raise RuntimeError(
                    f"Failed to load MobileNetV3 model from {args.model_path}: {summarize_exception(exc)}"
                ) from exc

    # If Roboflow credentials provided and YOLO usage requested, download weights
    if args.use_yolo and args.roboflow_api_key:
        try:
            from .roi_grabber_yolo import download_roboflow_weights
            yolo_path = download_roboflow_weights(
                api_key=args.roboflow_api_key,
                workspace=args.roboflow_workspace,
                project=args.roboflow_project,
                version=args.roboflow_version,
                target_dir=args.roboflow_download_dir,
            )
            logging.info("Downloaded Roboflow YOLO weights: %s", yolo_path)
            args.yolo_weights = yolo_path
        except Exception as exc:
            logging.warning("Roboflow download failed: %s", exc)

    if args.save_dir.exists():
        shutil.rmtree(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    model_input_size = (args.model_input_size, args.model_input_size)

    groups = sorted({sample.group_key for sample in sampled})
    
    # Log group distribution for debugging
    group_counts = {g: sum(1 for s in sampled if s.group_key == g) for g in groups}
    logging.info(f"Groups found in sampled data: {groups}")
    logging.info(f"Group sample counts: {group_counts}")
    
    active_group_idx = 0
    active_group = groups[active_group_idx]

    def filtered_indices(group_name: str) -> list[int]:
        return [idx for idx, sample in enumerate(sampled) if sample.group_key == group_name]

    visible = filtered_indices(active_group)
    if not visible:
        visible = list(range(len(sampled)))
        logging.warning(f"No samples found for group '{active_group}', showing all samples")
        active_group = "all"
    else:
        logging.info(f"Starting with group '{active_group}': {len(visible)} samples visible")

    visible_pos = 0
    threshold = float(args.threshold)
    
    # Print startup info
    print("\n" + "="*70)
    print("REVIEWER STARTUP INFO")
    print("="*70)
    print(f"Total samples: {len(sampled)}")
    print(f"Available groups ({len(groups)}): {', '.join(groups)}")
    for g in groups:
        count = sum(1 for s in sampled if s.group_key == g)
        print(f"  - {g}: {count} samples")
    print(f"\nStarting with group: {active_group} ({len(visible)} visible)")
    if args.interactive and not args.headless:
        print("\nKeyboard controls:")
        print("  n/d/→: Next image")
        print("  p/a/←: Previous image")
        print("  g: Cycle to next group")
        print("  +/-: Adjust MobileNetV3 threshold")
        print("  s: Save current overlay")
        print("  q/ESC: Quit")
    else:
        print(f"\nHeadless mode: saving all overlays to {args.save_dir}")
    print("="*70 + "\n")

    zip_cache: zipfile.ZipFile | None = None
    nested_zip_cache: dict[str, zipfile.ZipFile] = {}
    nested_zip_temp_paths: list[str] = []
    if use_zip:
        zip_cache = zipfile.ZipFile(args.zip_path, "r")
        for info in zip_cache.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".zip"):
                continue
            # Stream nested ZIPs to a temp file to avoid decompressing large members
            # directly into memory.
            with zip_cache.open(info.filename) as nested_fp:
                tmpf = tempfile.NamedTemporaryFile(delete=False)
                try:
                    shutil.copyfileobj(nested_fp, tmpf)
                    tmpf.close()
                    nested_zip_cache[info.filename] = zipfile.ZipFile(tmpf.name, "r")
                    nested_zip_temp_paths.append(tmpf.name)
                finally:
                    pass

    interactive_mode = args.interactive and not args.headless
    if interactive_mode:
        try:
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(args.window_name, 1400, 900)
        except cv2.error as exc:
            interactive_mode = False
            logging.warning(
                "OpenCV GUI support is unavailable; falling back to non-interactive export mode: %s",
                exc,
            )
    else:
        logging.info("Headless mode enabled: processing samples and saving overlays to %s", args.save_dir)
    
    if not interactive_mode:
        # In non-interactive mode, export all sampled items rather than just the
        # initially selected group so users get full coverage when GUI is missing.
        visible = list(range(len(sampled)))
        active_group = "all"
        logging.info("Non-interactive export mode: exporting %d sampled overlays", len(visible))

    try:
        while True:
            if not visible:
                visible = list(range(len(sampled)))
                active_group = "all"
                visible_pos = 0

            visible_pos = max(0, min(visible_pos, len(visible) - 1))
            sample_idx = visible[visible_pos]
            sample = sampled[sample_idx]

            image = load_sample_image(
                sample,
                zip_path=args.zip_path if use_zip else None,
                zip_cache=zip_cache,
                nested_zip_cache=nested_zip_cache if use_zip else None,
            )
            if image is None:
                logging.warning("Unreadable image: %s", sample.source_id)
                visible_pos = (visible_pos + 1) % max(1, len(visible))
                continue

            overlay, total_cells, parasites, parasitemia, variant_legend, roi_analysis_lines = compute_overlay(
                sample=sample,
                image=image,
                model=model,
                threshold=threshold,
                roi_size=args.roi_size,
                model_input_size=model_input_size,
                batch_size=args.batch_size,
                nms_iou_threshold=args.nms_iou_threshold,
                min_blob_area=args.min_blob_area,
                max_blob_area=args.max_blob_area,
                use_yolo=args.use_yolo,
                yolo_weights=args.yolo_weights,
                yolo_conf=args.yolo_conf,
                yolo_nms_iou=args.yolo_nms_iou,
                max_roi_analysis_lines=args.max_roi_analysis_lines,
                use_sahi=args.use_sahi_pipeline,
                sahi_tile_size=args.sahi_tile_size,
                sahi_overlap=args.sahi_overlap,
                sahi_nms_threshold=args.sahi_nms_threshold,
            )

            hud = annotate_hud(
                overlay=overlay,
                sample=sample,
                index=sample_idx,
                total=len(sampled),
                total_cells=total_cells,
                parasites=parasites,
                parasitemia=parasitemia,
                threshold=threshold,
                active_group=active_group,
                variant_legend=variant_legend,
                roi_analysis_lines=roi_analysis_lines[: max(0, args.max_roi_analysis_lines)],
            )
            if not interactive_mode:
                out_name = f"review_{sample_idx:05d}_{Path(sample.source_id).stem}.png"
                out_path = args.save_dir / out_name
                cv2.imwrite(str(out_path), hud)
                logging.info("Saved overlay: %s", out_path)
                if visible_pos >= len(visible) - 1:
                    break
                visible_pos += 1
                continue

            cv2.imshow(args.window_name, hud)

            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), ord("d"), 83):
                visible_pos = (visible_pos + 1) % len(visible)
                continue
            if key in (ord("p"), ord("a"), 81):
                visible_pos = (visible_pos - 1) % len(visible)
                continue
            if key == ord("g"):
                if groups:
                    active_group_idx = (active_group_idx + 1) % len(groups)
                    new_group = groups[active_group_idx]
                    visible = filtered_indices(new_group)
                    if visible:
                        active_group = new_group
                        visible_pos = 0
                        logging.info(f"Cycled to group '{new_group}': {len(visible)} samples visible")
                    else:
                        logging.warning(f"Group '{new_group}' is empty, staying on current group")
                        active_group_idx = (active_group_idx - 1) % len(groups)
                continue
            if key in (ord("+"), ord("=")):
                threshold = min(0.99, threshold + 0.01)
                continue
            if key in (ord("-"), ord("_")):
                threshold = max(0.01, threshold - 0.01)
                continue
            if key == ord("s"):
                out_name = f"review_{sample_idx:05d}_{Path(sample.source_id).stem}.png"
                out_path = args.save_dir / out_name
                cv2.imwrite(str(out_path), hud)
                logging.info("Saved overlay: %s", out_path)
                continue
    finally:
        for nested_archive in nested_zip_cache.values():
            nested_archive.close()
        if zip_cache is not None:
            zip_cache.close()
        for temp_path in nested_zip_temp_paths:
            try:
                os.remove(temp_path)
            except Exception:
                pass
        if interactive_mode:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()