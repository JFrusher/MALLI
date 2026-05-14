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
import random
import zipfile
import io

import cv2
import numpy as np
import tensorflow as tf

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
                        label_name=label,
                        smear_type=smear,
                    )
                )
                continue

            if ext != ".zip":
                continue

            nested_bytes = archive.read(info.filename)
            with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested:
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
                            label_name=label,
                            smear_type=smear,
                        )
                    )
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
    unknown = cv2.subtract(sure_bg, sure_fg_u8)

    _, markers = cv2.connectedComponents(sure_fg_u8)
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
        image_shape=image.shape,
        roi_size=roi_size,
        min_blob_area=min_blob_area,
        max_blob_area=max_blob_area,
    )
    return [VariantProposal(variant=variant, proposal=proposal) for proposal in proposals]


def draw_variant_overlay(
    image_bgr: np.ndarray,
    proposals: Sequence[VariantProposal],
    probabilities: np.ndarray | None,
    kept_indices: set[int],
    positive_indices: set[int],
) -> np.ndarray:
    overlay = image_bgr.copy()

    for idx, variant_proposal in enumerate(proposals):
        x1, y1, x2, y2 = variant_proposal.proposal.box
        score_text = ""
        if probabilities is not None and probabilities.size > idx:
            score_text = f" {probabilities[idx]:.2f}"

        label = f"{variant_proposal.variant.name}{score_text}"
        thickness = 4 if idx in positive_indices else 2 if idx in kept_indices else 1
        cv2.rectangle(overlay, (x1, y1), (x2, y2), variant_proposal.variant.color, thickness)
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            variant_proposal.variant.color,
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
) -> tuple[np.ndarray, int, int, float, list[str]]:
    variant_proposals: list[VariantProposal] = []
    raw_counts: Counter[str] = Counter()

    if use_yolo:
        try:
            from .roi_grabber_yolo import yolov8_detect_proposals
        except Exception:
            yolov8_detect_proposals = None

        if yolov8_detect_proposals is None or yolo_weights is None:
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
            # Use YOLO proposals and represent them as a single "yolo" variant
            boxes = yolov8_detect_proposals(
                image_bgr=image,
                yolo_weights=yolo_weights,
                conf_thresh=yolo_conf,
                nms_iou=yolo_nms_iou,
                roi_size=roi_size,
            )
            yolo_variant = SensingVariant("yolo", (0, 0, 255), 0, 0.0, 0.0, 0, 0, 0.0, 0)
            raw_counts[yolo_variant.name] = len(boxes)
            variant_proposals = [VariantProposal(variant=yolo_variant, proposal=prop) for prop in boxes]
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
        return image.copy(), 0, 0, 0.0, build_variant_legend(variants, raw_counts, Counter(), Counter(), model is not None)

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
        probabilities=probabilities,
        kept_indices=kept_indices,
        positive_indices=positive_indices,
    )

    total_cells = len(kept_indices_list)
    parasites = len(positive_indices)
    parasitemia = (100.0 * parasites / total_cells) if total_cells > 0 else 0.0
    legend_lines = build_variant_legend(variants, raw_counts, kept_counts, positive_counts, model is not None)
    return overlay, total_cells, parasites, parasitemia, legend_lines


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
) -> np.ndarray:
    canvas = overlay.copy()
    header_lines = [
        f"Image {index + 1}/{total} | group={sample.group_key} | active_group={active_group}",
        f"label={sample.label_name} | smear={sample.smear_type} | cells={total_cells} | parasites={parasites} | parasitemia={parasitemia:.2f}%",
        f"threshold={threshold:.2f} | keys: n/p next-prev, g cycle group, +/- threshold, s save, q quit",
    ]
    header_lines.extend(variant_legend)

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
    parser.add_argument("--model-path", type=Path, default=Path("models/last_mobilenetv3_small.keras"), help="Keras model path")
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
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    parser.add_argument("--use-yolo", action="store_true", help="Use YOLOv8 proposals instead of sensing variants")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLO weights path (default yolov8n.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.3, help="YOLO detection confidence threshold")
    parser.add_argument("--yolo-nms-iou", type=float, default=0.5, help="YOLO NMS IoU threshold for proposals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    source_samples: list[ReviewSample]
    use_zip = bool(args.use_zip)
    if not use_zip and args.dataset_root is not None and args.dataset_root.exists():
        source_samples = crawl_folder_samples(args.dataset_root, thick_min_dim=args.thick_min_dim)
    else:
        if not args.zip_path.exists():
            raise FileNotFoundError(
                f"ZIP path not found: {args.zip_path}. Provide --dataset-root or a valid --zip-path."
            )
        source_samples = crawl_zip_samples(args.zip_path, thick_min_dim=args.thick_min_dim)
        use_zip = True

    if not source_samples:
        raise RuntimeError("No review samples were discovered.")

    sampled = stratified_sample_by_group(source_samples, per_group=args.per_group, seed=args.seed)
    if not sampled:
        raise RuntimeError("No sampled items were selected. Increase --per-group.")

    model: tf.keras.Model | None = None
    if not args.disable_model:
        logging.info("Loading model: %s", args.model_path)
        model = tf.keras.models.load_model(str(args.model_path), compile=False)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    model_input_size = (args.model_input_size, args.model_input_size)

    groups = sorted({sample.group_key for sample in sampled})
    active_group_idx = 0
    active_group = groups[active_group_idx]

    def filtered_indices(group_name: str) -> list[int]:
        return [idx for idx, sample in enumerate(sampled) if sample.group_key == group_name]

    visible = filtered_indices(active_group)
    if not visible:
        visible = list(range(len(sampled)))
        active_group = "all"

    visible_pos = 0
    threshold = float(args.threshold)

    zip_cache: zipfile.ZipFile | None = None
    nested_zip_cache: dict[str, zipfile.ZipFile] = {}
    if use_zip:
        zip_cache = zipfile.ZipFile(args.zip_path, "r")
        for info in zip_cache.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".zip"):
                continue
            nested_bytes = zip_cache.read(info.filename)
            nested_zip_cache[info.filename] = zipfile.ZipFile(io.BytesIO(nested_bytes), "r")

    interactive_mode = True
    try:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, 1400, 900)
    except cv2.error as exc:
        interactive_mode = False
        logging.warning(
            "OpenCV GUI support is unavailable; falling back to non-interactive export mode: %s",
            exc,
        )

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

            overlay, total_cells, parasites, parasitemia, variant_legend = compute_overlay(
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
                    active_group = groups[active_group_idx]
                    visible = filtered_indices(active_group)
                    visible_pos = 0
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
        if interactive_mode:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()