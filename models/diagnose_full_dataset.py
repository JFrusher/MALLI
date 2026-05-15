from __future__ import annotations

"""Full-dataset diagnostic: run ROI grabber + MobileNetV3 on all images,
emit a CSV of per-image metrics and a set of diagnostic plots.

Usage example:
    python -m models.diagnose_full_dataset --zip-path dataverse_files.zip --model-path models/best_mobilenetv3_small.weights.h5 --output-dir outputs/diagnostics --write-csv
"""

from dataclasses import asdict
from pathlib import Path
from typing import List
import argparse
import logging
import math
import sys
import time
import tempfile
import shutil
import os
import zipfile

import numpy as np
import csv
import matplotlib.pyplot as plt

import tensorflow as tf

from .roi_grabber import (
    crawl_dataverse_zip,
    crawl_dataverse_structure,
    load_image_for_sample,
    preprocess_for_segmentation,
    segment_thick_smear_watershed,
    segment_thin_smear,
    extract_roi_proposals,
    batch_predict_rois,
    non_max_suppression,
)
from .inference import load_model_with_weights


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def format_eta(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def print_progress(current: int, total: int, started_at: float, prefix: str = "Progress") -> None:
    total = max(1, total)
    elapsed = max(0.0, time.perf_counter() - started_at)
    fraction = current / total
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = max(0.0, total - current)
    eta = remaining / rate if rate > 0 else float("inf")
    bar_width = 28
    filled = int(round(bar_width * fraction))
    bar = "█" * filled + "░" * (bar_width - filled)
    msg = (
        f"\r{prefix} [{bar}] {current}/{total} "
        f"({fraction * 100:5.1f}%) | {rate:5.2f} img/s | ETA {format_eta(eta)}"
    )
    print(msg, end="", file=sys.stderr, flush=True)
    if current >= total:
        print(file=sys.stderr, flush=True)


def process_all(
    samples,
    model: tf.keras.Model,
    threshold: float,
    roi_size: int,
    model_input_size: tuple[int, int],
    batch_size: int,
    nms_iou: float,
    min_blob_area: float,
    max_blob_area: float,
    zip_path: Path | None,
    zip_cache,
    nested_zip_cache,
    overlay_dir: Path | None = None,
    limit: int = 0,
    progress_total: int | None = None,
):
    rows = []
    count = 0
    started_at = time.perf_counter()
    total_for_progress = progress_total if progress_total is not None else (limit if limit else len(samples))
    processed_for_progress = 0
    for sample in samples:
        if limit and count >= limit:
            break
        count += 1
        processed_for_progress += 1
        img = load_image_for_sample(sample, zip_path=zip_path, zip_cache=zip_cache, nested_zip_cache=nested_zip_cache)
        if img is None:
            logging.warning("Skipping unreadable image: %s", sample.source_id)
            print_progress(processed_for_progress, total_for_progress, started_at)
            continue

        pre = preprocess_for_segmentation(img)
        if sample.smear_type == "thick":
            mask = segment_thick_smear_watershed(img, pre)
        else:
            mask = segment_thin_smear(pre)

        proposals = extract_roi_proposals(mask, image_shape=img.shape, roi_size=roi_size, min_blob_area=min_blob_area, max_blob_area=max_blob_area)
        proposal_areas = [p.area for p in proposals]
        raw_proposal_count = len(proposals)

        if raw_proposal_count == 0:
            avg_proposal_area = 0.0
            probabilities = np.zeros((0,), dtype=np.float32)
        else:
            avg_proposal_area = float(np.mean(proposal_areas))
            probabilities = batch_predict_rois(model=model, image_bgr=img, proposals=proposals, model_input_size=model_input_size, batch_size=batch_size)

        # Positive indices and NMS on positives to get kept parasite detections
        positive_mask = probabilities >= threshold if probabilities.size > 0 else np.array([], dtype=bool)
        positive_indices = np.where(positive_mask)[0] if probabilities.size > 0 else np.array([], dtype=int)
        kept_positive_set = set()
        if positive_indices.size > 0:
            pos_boxes = np.array([proposals[i].box for i in positive_indices], dtype=np.float32)
            pos_scores = probabilities[positive_indices]
            kept_rel = non_max_suppression(pos_boxes, pos_scores, iou_threshold=nms_iou)
            kept_positive_set = {int(positive_indices[r]) for r in kept_rel}

        total_cells = raw_proposal_count
        parasites = len(kept_positive_set)
        parasitemia = (100.0 * parasites / total_cells) if total_cells > 0 else 0.0

        avg_mnet_all = float(np.mean(probabilities)) if probabilities.size > 0 else 0.0
        avg_mnet_pos = float(np.mean(probabilities[positive_indices])) if positive_indices.size > 0 else 0.0
        median_mnet = float(np.median(probabilities)) if probabilities.size > 0 else 0.0
        std_mnet = float(np.std(probabilities)) if probabilities.size > 0 else 0.0

        group = f"{sample.smear_type}_{sample.label_name.lower()}"

        overlay_path = None
        if overlay_dir is not None and sample.smear_type == "thick":
            overlay_dir.mkdir(parents=True, exist_ok=True)
            # render a simple overlay similar to roi_grabber.draw_overlay
            from .roi_grabber import draw_overlay

            kept_positive_indices = kept_positive_set
            overlay = draw_overlay(img, proposals, probabilities, kept_positive_indices, threshold)
            stem = Path(sample.source_id).stem
            overlay_path = overlay_dir / f"{stem}_overlay.png"
            try:
                import cv2

                cv2.imwrite(str(overlay_path), overlay)
            except Exception:
                overlay_path = None

        rows.append(
            {
                "image_path": sample.source_id,
                "group": group,
                "label": sample.label_name,
                "smear": sample.smear_type,
                "total_cells": int(total_cells),
                "parasites": int(parasites),
                "parasitemia": float(parasitemia),
                "avg_proposal_area": float(avg_proposal_area),
                "raw_proposal_count": int(raw_proposal_count),
                "avg_mnet_all": float(avg_mnet_all),
                "avg_mnet_pos": float(avg_mnet_pos),
                "median_mnet": float(median_mnet),
                "std_mnet": float(std_mnet),
                "overlay_path": str(overlay_path) if overlay_path is not None else "",
            }
        )

        print_progress(processed_for_progress, total_for_progress, started_at)

    return rows


def summarize_and_plot(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Aggregate by group
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    stats_rows = []
    for grp, items in groups.items():
        paras = [it["parasitemia"] for it in items]
        totals = [it["total_cells"] for it in items]
        avg_mnets = [it["avg_mnet_all"] for it in items]
        avg_props = [it["avg_proposal_area"] for it in items]
        pct_zero = 100.0 * sum(1 for t in totals if t == 0) / len(totals)
        stats_rows.append(
            {
                "group": grp,
                "images": len(items),
                "mean_parasitemia": float(np.mean(paras)) if paras else 0.0,
                "median_parasitemia": float(np.median(paras)) if paras else 0.0,
                "mean_total_cells": float(np.mean(totals)) if totals else 0.0,
                "mean_avg_mnet": float(np.mean(avg_mnets)) if avg_mnets else 0.0,
                "mean_avg_proposal_area": float(np.mean(avg_props)) if avg_props else 0.0,
                "pct_zero_cells": float(pct_zero),
            }
        )
    # Write group stats CSV
    stats_path = out_dir / "group_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["group", "images", "mean_parasitemia", "median_parasitemia", "mean_total_cells", "mean_avg_mnet", "mean_avg_proposal_area", "pct_zero_cells"])
        writer.writeheader()
        for r in stats_rows:
            writer.writerow(r)

    # Plots using matplotlib only (avoid seaborn dependency)
    group_names = sorted(groups.keys())
    paras_by_group = [[r["parasitemia"] for r in groups[g]] for g in group_names]
    total_by_group = [[r["total_cells"] for r in groups[g]] for g in group_names]
    avg_mnet_by_group = [[r["avg_mnet_all"] for r in groups[g]] for g in group_names]
    raw_proposal_by_group = [[r["raw_proposal_count"] for r in groups[g]] for g in group_names]

    plt.figure(figsize=(10, 6))
    plt.boxplot(paras_by_group, labels=group_names)
    plt.xticks(rotation=45, ha="right")
    plt.title("Parasitemia distribution by group")
    plt.tight_layout()
    plt.savefig(out_dir / "parasitemia_by_group_box.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.boxplot(total_by_group, labels=group_names)
    plt.xticks(rotation=45, ha="right")
    plt.title("Detected cells per image by group")
    plt.tight_layout()
    plt.savefig(out_dir / "cells_per_image_by_group_box.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.boxplot(avg_mnet_by_group, labels=group_names)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average MobileNetV3 score per image by group")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_mnet_by_group_box.png")
    plt.close()

    # Scatter: parasitemia vs avg mnet, colored by group
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(group_names):
        grp_items = groups[g]
        x = [it["avg_mnet_all"] for it in grp_items]
        y = [it["parasitemia"] for it in grp_items]
        plt.scatter(x, y, color=cmap(i % 10), label=g, alpha=0.7)
    plt.xlabel("avg_mnet_all")
    plt.ylabel("parasitemia")
    plt.title("Parasitemia vs avg MobileNetV3 score (per image)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "parasitemia_vs_avg_mnet_scatter.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, g in enumerate(group_names):
        grp_items = groups[g]
        x = [it["raw_proposal_count"] for it in grp_items]
        y = [it["parasitemia"] for it in grp_items]
        plt.scatter(x, y, color=cmap(i % 10), label=g, alpha=0.7)
    plt.xlabel("raw_proposal_count")
    plt.ylabel("parasitemia")
    plt.title("Parasitemia vs proposal count (per image)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "parasitemia_vs_proposal_count.png")
    plt.close()

    # Save summary text
    txt = out_dir / "diagnostic_summary.txt"
    with txt.open("w", encoding="utf-8") as fh:
        fh.write("Group-level stats summary:\n")
        for r in stats_rows:
            fh.write(str(r) + "\n")
        fh.write("\nInvestigative notes:\n")
        # Simple heuristics to flag potential issues
        for r in stats_rows:
            notes = []
            if r["mean_total_cells"] < 5:
                notes.append("low detected cells (mean < 5)")
            if r["mean_avg_mnet"] < 0.3:
                notes.append("low classifier confidence (mean mnet < 0.3)")
            if r["pct_zero_cells"] > 50.0:
                notes.append("many images with zero detected cells (>50%)")
            fh.write(f"- {r['group']}: {', '.join(notes) if notes else 'no obvious flags'}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-dataset diagnostics for ROI grabber + MobileNetV3")
    parser.add_argument("--zip-path", type=Path, default=Path("dataverse_files.zip"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--use-zip", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path("models/best_mobilenetv3_small.weights.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--model-input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--min-blob-area", type=float, default=80.0)
    parser.add_argument("--max-blob-area", type=float, default=50000.0)
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overlay-thick", action="store_true", help="Save overlays for thick smears")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # Load model once using the configured input size.
    logging.info("Loading model: %s", args.model_path)
    if args.model_path.suffix.lower() == ".weights.h5" or args.model_path.name.endswith(".weights.h5"):
        model = load_model_with_weights(args.model_path, input_shape=(args.model_input_size, args.model_input_size, 3), compile_model=False)
    else:
        try:
            model = tf.keras.models.load_model(str(args.model_path), compile=False, safe_mode=False)
        except TypeError:
            model = tf.keras.models.load_model(str(args.model_path), compile=False)

    if args.dataset_root is not None and args.dataset_root.exists() and not args.use_zip:
        samples = crawl_dataverse_structure(args.dataset_root, thick_min_dim=512)
        zip_path = None
        zip_cache = None
        nested_zip_cache = None
    else:
        if not args.zip_path.exists():
            raise FileNotFoundError(f"ZIP not found: {args.zip_path}")
        samples = crawl_dataverse_zip(args.zip_path, thick_min_dim=512)
        zip_path = args.zip_path
        zip_cache = zipfile.ZipFile(args.zip_path, "r")
        nested_zip_cache = {}
        temp_paths: list[str] = []
        for info in zip_cache.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".zip"):
                continue
            with zip_cache.open(info.filename) as nested_fp:
                tmpf = tempfile.NamedTemporaryFile(delete=False)
                try:
                    shutil.copyfileobj(nested_fp, tmpf)
                    tmpf.close()
                    nested_zip_cache[info.filename] = zipfile.ZipFile(tmpf.name, "r")
                    temp_paths.append(tmpf.name)
                finally:
                    try:
                        tmpf.close()
                    except Exception:
                        pass

    try:
        logging.info("Discovered %d images to process", len(samples))

        overlay_dir = args.output_dir / "overlays" if args.overlay_thick else None

        df = process_all(
            samples=samples,
            model=model,
            threshold=args.threshold,
            roi_size=args.roi_size,
            model_input_size=(args.model_input_size, args.model_input_size),
            batch_size=args.batch_size,
            nms_iou=args.nms_iou,
            min_blob_area=args.min_blob_area,
            max_blob_area=args.max_blob_area,
            zip_path=zip_path,
            zip_cache=zip_cache,
            nested_zip_cache=nested_zip_cache,
            overlay_dir=overlay_dir,
            limit=args.limit,
            progress_total=min(args.limit, len(samples)) if args.limit else len(samples),
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.write_csv:
            csv_path = args.output_dir / "diagnostics_per_image.csv"
            # df is a list of dict rows
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                if df:
                    writer = csv.DictWriter(fh, fieldnames=list(df[0].keys()))
                    writer.writeheader()
                    for r in df:
                        writer.writerow(r)
            logging.info("Wrote CSV: %s", csv_path)

        # Run analysis and plots
        summarize_and_plot(df, args.output_dir)
        logging.info("Diagnostics complete. Outputs written to %s", args.output_dir)
    finally:
        if zip_cache is not None:
            try:
                zip_cache.close()
            except Exception:
                pass
        for nested in nested_zip_cache.values():
            try:
                nested.close()
            except Exception:
                pass
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
