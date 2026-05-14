from __future__ import annotations

"""Region Proposal (ROI) grabber for full-field malaria diagnostics.

This script crawls an extracted Dataverse-style dataset, proposes cell ROIs from
microscopy full-field images, runs batch inference with a pre-trained
MobileNetV3 classifier, applies NMS to de-duplicate overlapping parasite
detections, and writes overlays and summary metrics.

Example:
    python -m models.roi_grabber --dataset-root nih_data --output-dir outputs/roi_grabber
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import argparse
import csv
import gzip
import io
import json
import logging
import re
import zipfile

import cv2
import numpy as np
import tensorflow as tf


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    source_type: str
    source_id: str
    label_name: str
    smear_type: str


@dataclass(frozen=True)
class RoiProposal:
    box: tuple[int, int, int, int]
    centroid: tuple[int, int]
    area: float


@dataclass(frozen=True)
class ImageResult:
    image_path: str
    label_name: str
    smear_type: str
    total_cells_detected: int
    parasites_identified: int
    parasitemia_percent: float
    overlay_path: Path | None


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _parts_to_tokens(parts_lower: Sequence[str]) -> set[str]:
    tokens: set[str] = set()
    for part in parts_lower:
        for token in re.split(r"[^a-z0-9]+", part.lower()):
            if token:
                tokens.add(token)
    return tokens


def infer_label_from_parts(parts_lower: Sequence[str]) -> str | None:
    tokens = _parts_to_tokens(parts_lower)
    if {"uninfected", "negative", "healthy"}.intersection(tokens):
        return "Uninfected"
    if {"infected", "parasitized", "positive"}.intersection(tokens):
        return "Infected"
    return None


def infer_smear_type_from_parts(parts_lower: Sequence[str]) -> str | None:
    tokens = _parts_to_tokens(parts_lower)
    if "thick" in tokens:
        return "thick"
    if "thin" in tokens:
        return "thin"
    return None


def infer_smear_type_by_dimensions(height: int, width: int, thick_min_dim: int = 512) -> str:
    return "thick" if min(height, width) >= thick_min_dim else "thin"


def read_image_from_zip(archive: zipfile.ZipFile, member_name: str) -> np.ndarray | None:
    with archive.open(member_name, "r") as handle:
        raw = handle.read()
    image_u8 = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(image_u8, cv2.IMREAD_COLOR)


def make_nested_source_id(outer_member: str, inner_member: str) -> str:
    return f"{outer_member}::{inner_member}"


def split_nested_source_id(source_id: str) -> tuple[str, str] | None:
    if "::" not in source_id:
        return None
    outer, inner = source_id.split("::", 1)
    return outer, inner


def crawl_dataverse_structure(dataset_root: Path, thick_min_dim: int = 512) -> list[ImageSample]:
    """Crawl extracted dataset folders and infer class + smear type per image."""

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples: list[ImageSample] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        parts_lower = [part.lower() for part in path.parts]
        label = infer_label_from_parts(parts_lower)
        if label is None:
            continue

        smear_type = infer_smear_type_from_parts(parts_lower)
        if smear_type is None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                logging.warning("Skipping unreadable image during crawl: %s", path)
                continue
            h, w = image.shape[:2]
            smear_type = infer_smear_type_by_dimensions(h, w, thick_min_dim=thick_min_dim)

        samples.append(
            ImageSample(
                source_type="folder",
                source_id=str(path),
                label_name=label,
                smear_type=smear_type,
            )
        )

    if not samples:
        raise ValueError(
            "No valid images found under dataset root. Expected folders containing "
            "Infected/Uninfected or Parasitized/Uninfected images."
        )

    logging.info("Discovered %d images for processing", len(samples))
    return samples


def crawl_dataverse_zip(zip_path: Path, thick_min_dim: int = 512) -> list[ImageSample]:
    """Crawl Dataverse ZIP layouts, including ZIP-of-ZIPs structures."""

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP path not found: {zip_path}")

    samples: list[ImageSample] = []
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

                smear_type = infer_smear_type_from_parts(parts_lower)
                if smear_type is None:
                    image = read_image_from_zip(archive, info.filename)
                    if image is None:
                        continue
                    h, w = image.shape[:2]
                    smear_type = infer_smear_type_by_dimensions(h, w, thick_min_dim=thick_min_dim)

                samples.append(
                    ImageSample(
                        source_type="zip",
                        source_id=info.filename,
                        label_name=label,
                        smear_type=smear_type,
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

                    smear_type = infer_smear_type_from_parts(combined_parts)
                    if smear_type is None:
                        image = read_image_from_zip(nested, nested_info.filename)
                        if image is None:
                            continue
                        h, w = image.shape[:2]
                        smear_type = infer_smear_type_by_dimensions(h, w, thick_min_dim=thick_min_dim)

                    samples.append(
                        ImageSample(
                            source_type="nested_zip",
                            source_id=make_nested_source_id(info.filename, nested_info.filename),
                            label_name=label,
                            smear_type=smear_type,
                        )
                    )

    if not samples:
        raise ValueError("No image samples were discovered in the ZIP archive.")

    logging.info("Discovered %d images for processing from zip source", len(samples))
    return samples


def load_image_for_sample(
    sample: ImageSample,
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

    if zip_cache is not None:
        return read_image_from_zip(zip_cache, sample.source_id)

    with zipfile.ZipFile(zip_path, "r") as archive:
        return read_image_from_zip(archive, sample.source_id)


def preprocess_for_segmentation(image_bgr: np.ndarray) -> np.ndarray:
    """Apply Foldscope-aware contrast normalization and denoising."""

    green = image_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(green)
    blurred = cv2.GaussianBlur(normalized, (5, 5), sigmaX=0)
    return blurred


def segment_thick_smear_watershed(image_bgr: np.ndarray, preprocessed: np.ndarray) -> np.ndarray:
    """Segment de-clumped candidates in thick smear images using watershed."""

    _, thresholded = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, 0.35 * distance.max(), 255, 0)
    sure_fg_u8 = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg_u8)

    _, markers = cv2.connectedComponents(sure_fg_u8)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_markers = cv2.watershed(image_bgr.copy(), markers)
    mask = np.zeros(preprocessed.shape, dtype=np.uint8)
    mask[watershed_markers > 1] = 255
    return mask


def segment_thin_smear(preprocessed: np.ndarray) -> np.ndarray:
    """Segment cell-like foreground in thin smear images."""

    _, thresholded = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.medianBlur(opened, 3)
    return cleaned


def compute_square_box(
    centroid_x: int,
    centroid_y: int,
    roi_size: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    half = roi_size // 2
    x1 = centroid_x - half
    y1 = centroid_y - half
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        shift = x2 - width
        x1 = max(0, x1 - shift)
        x2 = width
    if y2 > height:
        shift = y2 - height
        y1 = max(0, y1 - shift)
        y2 = height

    return x1, y1, x2, y2


def extract_roi_proposals(
    cell_mask: np.ndarray,
    image_shape: tuple[int, int, int],
    roi_size: int = 128,
    min_blob_area: float = 80.0,
    max_blob_area: float = 50_000.0,
) -> list[RoiProposal]:
    """Extract centroid-centered square ROI proposals from contour blobs."""

    height, width = image_shape[:2]
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals: list[RoiProposal] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_blob_area or area > max_blob_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        box = compute_square_box(cx, cy, roi_size=roi_size, width=width, height=height)
        proposals.append(RoiProposal(box=box, centroid=(cx, cy), area=area))

    return proposals


def crop_and_prepare_roi(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    model_input_size: tuple[int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Encountered empty ROI crop.")

    resized = cv2.resize(crop, model_input_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def batch_predict_rois(
    model: tf.keras.Model,
    image_bgr: np.ndarray,
    proposals: Sequence[RoiProposal],
    model_input_size: tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    if not proposals:
        return np.zeros((0,), dtype=np.float32)

    roi_batch = np.stack(
        [crop_and_prepare_roi(image_bgr, proposal.box, model_input_size) for proposal in proposals],
        axis=0,
    )
    probabilities = model.predict(roi_batch, batch_size=batch_size, verbose=0).reshape(-1)
    return probabilities.astype(np.float32)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Return kept indices after score-ordered NMS."""

    if boxes.size == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = np.array([compute_iou(boxes[idx], boxes[other]) for other in remaining], dtype=np.float32)
        order = remaining[ious < iou_threshold]

    return keep


def draw_overlay(
    image_bgr: np.ndarray,
    proposals: Sequence[RoiProposal],
    probabilities: np.ndarray,
    kept_positive_indices: set[int],
    threshold: float,
) -> np.ndarray:
    overlay = image_bgr.copy()

    for idx, proposal in enumerate(proposals):
        x1, y1, x2, y2 = proposal.box
        if idx in kept_positive_indices:
            color = (0, 0, 255)
            label_text = f"INF {probabilities[idx]:.2f}"
        elif probabilities[idx] >= threshold:
            continue
        else:
            color = (0, 255, 0)
            label_text = f"HLT {probabilities[idx]:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            label_text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay


def process_single_image(
    sample: ImageSample,
    model: tf.keras.Model,
    threshold: float,
    roi_size: int,
    model_input_size: tuple[int, int],
    batch_size: int,
    nms_iou_threshold: float,
    min_blob_area: float,
    max_blob_area: float,
    overlay_dir: Path,
    zip_path: Path | None,
    zip_cache: zipfile.ZipFile | None,
    nested_zip_cache: dict[str, zipfile.ZipFile] | None,
    yolo_weights: str | None = None,
    yolo_conf: float = 0.3,
    yolo_nms_iou: float = 0.5,
) -> ImageResult | None:
    image = load_image_for_sample(
        sample=sample,
        zip_path=zip_path,
        zip_cache=zip_cache,
        nested_zip_cache=nested_zip_cache,
    )
    if image is None:
        logging.warning("Could not read image: %s", sample.source_id)
        return None

    # Optionally use YOLO-based proposals instead of segmentation-based proposals
    proposals: list[RoiProposal]
    if yolo_weights:
        try:
            from .roi_grabber_yolo import yolov8_detect_proposals

            proposals = yolov8_detect_proposals(
                image_bgr=image,
                yolo_weights=yolo_weights,
                conf_thresh=yolo_conf,
                nms_iou=yolo_nms_iou,
                roi_size=roi_size,
            )
        except Exception as exc:
            logging.warning("YOLO proposals failed; falling back to segmentation: %s", exc)
            preprocessed = preprocess_for_segmentation(image)
            if sample.smear_type == "thick":
                mask = segment_thick_smear_watershed(image, preprocessed)
            else:
                mask = segment_thin_smear(preprocessed)

            proposals = extract_roi_proposals(
                mask,
                image_shape=image.shape,
                roi_size=roi_size,
                min_blob_area=min_blob_area,
                max_blob_area=max_blob_area,
            )
    else:
        preprocessed = preprocess_for_segmentation(image)
        if sample.smear_type == "thick":
            mask = segment_thick_smear_watershed(image, preprocessed)
        else:
            mask = segment_thin_smear(preprocessed)

        proposals = extract_roi_proposals(
            mask,
            image_shape=image.shape,
            roi_size=roi_size,
            min_blob_area=min_blob_area,
            max_blob_area=max_blob_area,
        )

    if not proposals:
        return ImageResult(
            image_path=sample.source_id,
            label_name=sample.label_name,
            smear_type=sample.smear_type,
            total_cells_detected=0,
            parasites_identified=0,
            parasitemia_percent=0.0,
            overlay_path=None,
        )

    probabilities = batch_predict_rois(
        model=model,
        image_bgr=image,
        proposals=proposals,
        model_input_size=model_input_size,
        batch_size=batch_size,
    )

    positive_indices = np.where(probabilities >= threshold)[0]
    kept_positive_set: set[int] = set()
    if positive_indices.size > 0:
        positive_boxes = np.array([proposals[idx].box for idx in positive_indices], dtype=np.float32)
        positive_scores = probabilities[positive_indices]
        kept_positive_rel = non_max_suppression(positive_boxes, positive_scores, iou_threshold=nms_iou_threshold)
        kept_positive_set = {int(positive_indices[rel_idx]) for rel_idx in kept_positive_rel}

    total_cells = len(proposals)
    parasites = len(kept_positive_set)
    parasitemia = (100.0 * parasites / total_cells) if total_cells > 0 else 0.0

    overlay_path: Path | None = None
    if sample.smear_type == "thick":
        overlay = draw_overlay(
            image_bgr=image,
            proposals=proposals,
            probabilities=probabilities,
            kept_positive_indices=kept_positive_set,
            threshold=threshold,
        )
        overlay_stem = Path(sample.source_id).stem
        overlay_path = overlay_dir / f"{overlay_stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

    return ImageResult(
        image_path=sample.source_id,
        label_name=sample.label_name,
        smear_type=sample.smear_type,
        total_cells_detected=total_cells,
        parasites_identified=parasites,
        parasitemia_percent=parasitemia,
        overlay_path=overlay_path,
    )


def _write_sharded_jsonl_gz(results: Sequence[ImageResult], report_dir: Path, shard_size: int) -> None:
    shard_dir = report_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    effective_shard_size = max(1, int(shard_size))
    shard_count = (len(results) + effective_shard_size - 1) // effective_shard_size
    for shard_index in range(shard_count):
        start = shard_index * effective_shard_size
        end = min(len(results), start + effective_shard_size)
        shard_path = shard_dir / f"part-{shard_index:05d}-of-{shard_count:05d}.jsonl.gz"
        with gzip.open(shard_path, "wt", encoding="utf-8", newline="\n") as handle:
            for result in results[start:end]:
                row = {
                    "image_path": result.image_path,
                    "label_name": result.label_name,
                    "smear_type": result.smear_type,
                    "total_cells_detected": result.total_cells_detected,
                    "parasites_identified": result.parasites_identified,
                    "parasitemia_percent": result.parasitemia_percent,
                    "overlay_path": str(result.overlay_path) if result.overlay_path else "",
                }
                handle.write(json.dumps(row) + "\n")


def write_reports(
    results: Sequence[ImageResult],
    report_dir: Path,
    shard_size: int,
    write_csv: bool,
) -> dict[str, float | int]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "roi_summary_global.json"
    _write_sharded_jsonl_gz(results, report_dir=report_dir, shard_size=shard_size)

    total_cells = int(sum(result.total_cells_detected for result in results))
    total_parasites = int(sum(result.parasites_identified for result in results))
    parasitemia = (100.0 * total_parasites / total_cells) if total_cells > 0 else 0.0

    if write_csv:
        csv_path = report_dir / "roi_summary_per_image.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "image_path",
                    "label_name",
                    "smear_type",
                    "total_cells_detected",
                    "parasites_identified",
                    "parasitemia_percent",
                    "overlay_path",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.image_path,
                        result.label_name,
                        result.smear_type,
                        result.total_cells_detected,
                        result.parasites_identified,
                        f"{result.parasitemia_percent:.4f}",
                        str(result.overlay_path) if result.overlay_path else "",
                    ]
                )

    summary = {
        "images_processed": len(results),
        "total_cells_detected": total_cells,
        "total_parasites_identified": total_parasites,
        "parasitemia_percent": float(parasitemia),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI grabber and full-field malaria inference")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to extracted Dataverse dataset root")
    parser.add_argument("--zip-path", type=Path, default=Path("dataverse_files.zip"), help="Path to Dataverse ZIP")
    parser.add_argument("--use-zip", action="store_true", help="Force reading image data from ZIP")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/last_mobilenetv3_small.keras"),
        help="Path to trained Keras model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/roi_grabber"),
        help="Output directory for overlays and reports",
    )
    parser.add_argument("--threshold", type=float, default=0.85, help="Parasite-positive probability threshold")
    parser.add_argument("--roi-size", type=int, default=128, help="Square ROI crop size around each blob centroid")
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=224,
        help="Classifier input edge length (ROI crops are resized to this)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.30, help="IoU threshold used in NMS")
    parser.add_argument("--thick-min-dim", type=int, default=512, help="Dimension fallback for thick smear inference")
    parser.add_argument("--min-blob-area", type=float, default=80.0, help="Minimum contour area to keep")
    parser.add_argument("--max-blob-area", type=float, default=50000.0, help="Maximum contour area to keep")
    parser.add_argument("--report-shard-size", type=int, default=2000, help="Rows per compressed report shard")
    parser.add_argument("--write-csv", action="store_true", help="Also write an uncompressed per-image CSV")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to process (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--use-yolo", action="store_true", help="Use YOLOv8 proposals instead of segmentation")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt", help="YOLO weights path (default yolov8n.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.3, help="YOLO detection confidence threshold")
    parser.add_argument("--yolo-nms-iou", type=float, default=0.5, help="YOLO NMS IoU threshold for proposals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    overlay_dir = args.output_dir / "overlays"
    report_dir = args.output_dir / "reports"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    use_zip = bool(args.use_zip)
    if not use_zip and args.dataset_root is not None and args.dataset_root.exists():
        samples = crawl_dataverse_structure(args.dataset_root, thick_min_dim=args.thick_min_dim)
    else:
        samples = crawl_dataverse_zip(args.zip_path, thick_min_dim=args.thick_min_dim)
        use_zip = True

    if args.limit > 0:
        samples = samples[: args.limit]

    logging.info("Loading model: %s", args.model_path)
    model = tf.keras.models.load_model(str(args.model_path), compile=False)

    model_input_size = (args.model_input_size, args.model_input_size)
    results: list[ImageResult] = []

    zip_cache: zipfile.ZipFile | None = None
    nested_zip_cache: dict[str, zipfile.ZipFile] = {}
    if use_zip:
        zip_cache = zipfile.ZipFile(args.zip_path, "r")
        for info in zip_cache.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".zip"):
                continue
            nested_bytes = zip_cache.read(info.filename)
            nested_zip_cache[info.filename] = zipfile.ZipFile(io.BytesIO(nested_bytes), "r")

    try:
        for index, sample in enumerate(samples, start=1):
            result = process_single_image(
                sample=sample,
                model=model,
                threshold=args.threshold,
                roi_size=args.roi_size,
                model_input_size=model_input_size,
                batch_size=args.batch_size,
                nms_iou_threshold=args.nms_iou_threshold,
                min_blob_area=args.min_blob_area,
                max_blob_area=args.max_blob_area,
                overlay_dir=overlay_dir,
                zip_path=args.zip_path if use_zip else None,
                zip_cache=zip_cache,
                nested_zip_cache=nested_zip_cache if use_zip else None,
                yolo_weights=args.yolo_weights if args.use_yolo else None,
                yolo_conf=args.yolo_conf,
                yolo_nms_iou=args.yolo_nms_iou,
            )
            if result is not None:
                results.append(result)

            if index % 50 == 0:
                logging.info("Processed %d/%d images", index, len(samples))
    finally:
        for nested_archive in nested_zip_cache.values():
            nested_archive.close()
        if zip_cache is not None:
            zip_cache.close()

    if not results:
        raise RuntimeError("No images were processed successfully.")

    summary = write_reports(
        results,
        report_dir=report_dir,
        shard_size=args.report_shard_size,
        write_csv=args.write_csv,
    )

    logging.info("Processing complete")
    logging.info("Images processed: %d", int(summary["images_processed"]))
    logging.info("Total cells detected: %d", int(summary["total_cells_detected"]))
    logging.info("Total parasites identified: %d", int(summary["total_parasites_identified"]))
    logging.info("Parasitemia percentage: %.4f", float(summary["parasitemia_percent"]))
    logging.info("Reports directory: %s", report_dir)


if __name__ == "__main__":
    main()