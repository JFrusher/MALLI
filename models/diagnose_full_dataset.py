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
import json
import logging
import math
import os
import sys
import time
import tempfile
import shutil
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
from .model_factory import build_mobilenetv3_small


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


def cohen_d(sample_a: list[float], sample_b: list[float]) -> float:
    if not sample_a or not sample_b:
        return 0.0
    mean_a = float(np.mean(sample_a))
    mean_b = float(np.mean(sample_b))
    var_a = float(np.var(sample_a, ddof=1)) if len(sample_a) > 1 else 0.0
    var_b = float(np.var(sample_b, ddof=1)) if len(sample_b) > 1 else 0.0
    pooled_num = (len(sample_a) - 1) * var_a + (len(sample_b) - 1) * var_b
    pooled_den = max(1, len(sample_a) + len(sample_b) - 2)
    pooled_var = pooled_num / pooled_den
    denom = math.sqrt(max(1e-12, pooled_var))
    return (mean_a - mean_b) / denom


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def build_summary_row(items: list[dict], name: str) -> dict[str, float | int | str]:
    paras = [float(it["parasitemia"]) for it in items]
    totals = [int(it["total_cells"]) for it in items]
    parasites = [int(it["parasites"]) for it in items]
    avg_mnets = [float(it["avg_mnet_all"]) for it in items]
    avg_pos_mnets = [float(it["avg_mnet_pos"]) for it in items if float(it["avg_mnet_pos"]) > 0.0]
    avg_props = [float(it["avg_proposal_area"]) for it in items]
    raw_counts = [int(it["raw_proposal_count"]) for it in items]

    para_pct = percentile_summary(paras)
    total_pct = percentile_summary([float(v) for v in totals])
    raw_pct = percentile_summary([float(v) for v in raw_counts])

    zero_cells = sum(1 for t in totals if t == 0)
    zero_parasitemia = sum(1 for p in paras if p == 0.0)
    any_parasites = sum(1 for p in parasites if p > 0)

    return {
        "group": name,
        "images": len(items),
        "mean_parasitemia": float(np.mean(paras)) if paras else 0.0,
        "median_parasitemia": para_pct["median"],
        "std_parasitemia": float(np.std(paras)) if len(paras) > 1 else 0.0,
        "min_parasitemia": para_pct["min"],
        "q25_parasitemia": para_pct["q25"],
        "q75_parasitemia": para_pct["q75"],
        "max_parasitemia": para_pct["max"],
        "mean_total_cells": float(np.mean(totals)) if totals else 0.0,
        "median_total_cells": total_pct["median"],
        "mean_parasites": float(np.mean(parasites)) if parasites else 0.0,
        "mean_avg_mnet": float(np.mean(avg_mnets)) if avg_mnets else 0.0,
        "median_avg_mnet": float(np.median(avg_mnets)) if avg_mnets else 0.0,
        "mean_avg_mnet_pos": float(np.mean(avg_pos_mnets)) if avg_pos_mnets else 0.0,
        "mean_avg_proposal_area": float(np.mean(avg_props)) if avg_props else 0.0,
        "mean_raw_proposal_count": float(np.mean(raw_counts)) if raw_counts else 0.0,
        "median_raw_proposal_count": raw_pct["median"],
        "pct_zero_cells": 100.0 * zero_cells / len(items) if items else 0.0,
        "pct_zero_parasitemia": 100.0 * zero_parasitemia / len(items) if items else 0.0,
        "pct_any_parasites": 100.0 * any_parasites / len(items) if items else 0.0,
    }


def infer_input_size_from_keras_archive(model_path: Path) -> tuple[int, int] | None:
    if not model_path.exists() or not zipfile.is_zipfile(model_path):
        return None

    with zipfile.ZipFile(model_path, "r") as archive:
        try:
            config = json.loads(archive.read("config.json"))
        except Exception:
            return None

    layers = config.get("config", {}).get("layers", [])
    for layer in layers:
        if layer.get("class_name") != "InputLayer":
            continue
        batch_shape = layer.get("config", {}).get("batch_input_shape")
        if isinstance(batch_shape, list) and len(batch_shape) >= 3:
            return int(batch_shape[1]), int(batch_shape[2])
    return None


def load_diagnostic_model(model_path: Path, fallback_input_size: tuple[int, int]) -> tuple[tf.keras.Model, tuple[int, int]]:
    if model_path.suffix.lower() == ".weights.h5" or model_path.name.endswith(".weights.h5"):
        model = load_model_with_weights(
            model_path,
            input_shape=(fallback_input_size[0], fallback_input_size[1], 3),
            compile_model=False,
        )
        return model, fallback_input_size

    archive_input_size = infer_input_size_from_keras_archive(model_path)
    input_size = archive_input_size or fallback_input_size

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
        return model, input_size
    except Exception as exc:
        logging.warning("Direct Keras loading failed for %s: %s", model_path, exc)

    if zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path, "r") as archive:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5") as tmpf:
                    tmpf.write(archive.read("model.weights.h5"))
                    weights_path = Path(tmpf.name)
            except Exception as exc:
                raise RuntimeError(f"Could not extract embedded weights from {model_path}: {exc}") from exc

        try:
            model = build_mobilenetv3_small(input_shape=(input_size[0], input_size[1], 3))
            # The weights inside the Keras archive are name-matched against the
            # rebuilt architecture to avoid the trackable-loading bug in tf_keras.
            try:
                model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
            except TypeError:
                model.load_weights(str(weights_path))
            return model, input_size
        finally:
            try:
                os.remove(weights_path)
            except Exception:
                pass

    raise RuntimeError(f"Unsupported model format: {model_path}")


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
    # Aggregate by views that answer the actual evaluation questions.
    groups: dict[str, list[dict]] = {}
    labels: dict[str, list[dict]] = {}
    smears: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
        labels.setdefault(r["label"], []).append(r)
        smears.setdefault(r["smear"], []).append(r)

    overall_stats = build_summary_row(rows, "all")
    group_stats = [build_summary_row(items, grp) for grp, items in sorted(groups.items())]
    label_stats = [build_summary_row(items, label) for label, items in sorted(labels.items())]
    smear_stats = [build_summary_row(items, smear) for smear, items in sorted(smears.items())]

    # Write CSV breakdowns.
    def write_csv(path: Path, payload: list[dict[str, float | int | str]]) -> None:
        if not payload:
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(payload[0].keys()))
            writer.writeheader()
            for row in payload:
                writer.writerow(row)

    write_csv(out_dir / "group_stats.csv", group_stats)
    write_csv(out_dir / "label_stats.csv", label_stats)
    write_csv(out_dir / "smear_stats.csv", smear_stats)

    # Plots using matplotlib only.
    def boxplot_by_key(key: str, title: str, file_name: str, order: list[str]) -> None:
        series = [[r["parasitemia"] for r in (groups.get(k) if key == "group" else labels.get(k) if key == "label" else smears.get(k))] for k in order]
        plt.figure(figsize=(max(10, 0.8 * len(order)), 6))
        plt.boxplot(series, labels=order, showmeans=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("parasitemia")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / file_name)
        plt.close()

    def hist_by_key(key: str, title: str, file_name: str, order: list[str]) -> None:
        plt.figure(figsize=(11, 6))
        bins = np.linspace(0, max(1.0, float(max((r["parasitemia"] for r in rows), default=0.0))), 20)
        alpha = 0.45
        cmap = plt.get_cmap("tab10")
        for i, k in enumerate(order):
            series = [r["parasitemia"] for r in (groups.get(k) if key == "group" else labels.get(k) if key == "label" else smears.get(k))]
            if not series:
                continue
            plt.hist(series, bins=bins, alpha=alpha, color=cmap(i % 10), label=k, density=True)
        plt.xlabel("parasitemia")
        plt.ylabel("density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / file_name)
        plt.close()

    group_names = sorted(groups.keys())
    label_names = sorted(labels.keys())
    smear_names = sorted(smears.keys())

    boxplot_by_key("group", "Parasitemia distribution by smear/label group", "parasitemia_by_group_box.png", group_names)
    boxplot_by_key("label", "Parasitemia distribution by label", "parasitemia_by_label_box.png", label_names)
    boxplot_by_key("smear", "Parasitemia distribution by smear type", "parasitemia_by_smear_box.png", smear_names)

    hist_by_key("group", "Parasitemia density by group", "parasitemia_by_group_hist.png", group_names)
    hist_by_key("label", "Parasitemia density by label", "parasitemia_by_label_hist.png", label_names)
    hist_by_key("smear", "Parasitemia density by smear type", "parasitemia_by_smear_hist.png", smear_names)

    # Cell-count and score plots.
    plt.figure(figsize=(10, 6))
    plt.boxplot([[r["total_cells"] for r in groups[g]] for g in group_names], labels=group_names, showmeans=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("total cells detected")
    plt.title("Detected cells per image by group")
    plt.tight_layout()
    plt.savefig(out_dir / "cells_per_image_by_group_box.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.boxplot([[r["avg_mnet_all"] for r in groups[g]] for g in group_names], labels=group_names, showmeans=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("avg_mnet_all")
    plt.title("Average MobileNetV3 score per image by group")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_mnet_by_group_box.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(group_names):
        grp_items = groups[g]
        x = [it["avg_mnet_all"] for it in grp_items]
        y = [it["parasitemia"] for it in grp_items]
        plt.scatter(x, y, color=cmap(i % 10), label=g, alpha=0.75, s=36)
    plt.xlabel("avg_mnet_all")
    plt.ylabel("parasitemia")
    plt.title("Parasitemia vs avg MobileNetV3 score (per image)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "parasitemia_vs_avg_mnet_scatter.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, g in enumerate(group_names):
        grp_items = groups[g]
        x = [it["raw_proposal_count"] for it in grp_items]
        y = [it["parasitemia"] for it in grp_items]
        plt.scatter(x, y, color=cmap(i % 10), label=g, alpha=0.75, s=36)
    plt.xlabel("raw_proposal_count")
    plt.ylabel("parasitemia")
    plt.title("Parasitemia vs proposal count (per image)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "parasitemia_vs_proposal_count.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, smear in enumerate(smear_names):
        grp_items = smears[smear]
        x = [it["total_cells"] for it in grp_items]
        y = [it["parasitemia"] for it in grp_items]
        plt.scatter(x, y, color=cmap(i % 10), label=smear, alpha=0.75, s=36)
    plt.xlabel("total_cells")
    plt.ylabel("parasitemia")
    plt.title("Parasitemia vs detected cell count by smear type")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "parasitemia_vs_cells_by_smear.png")
    plt.close()

    # Storytelling summary text with effect sizes and flags.
    infected = labels.get("Infected", [])
    uninfected = labels.get("Uninfected", [])
    thick = smears.get("thick", [])
    thin = smears.get("thin", [])

    infected_paras = [float(r["parasitemia"]) for r in infected]
    uninfected_paras = [float(r["parasitemia"]) for r in uninfected]
    thick_paras = [float(r["parasitemia"]) for r in thick]
    thin_paras = [float(r["parasitemia"]) for r in thin]

    txt = out_dir / "diagnostic_summary.txt"
    with txt.open("w", encoding="utf-8") as fh:
        fh.write("Overall summary:\n")
        fh.write(str(overall_stats) + "\n\n")
        fh.write("Interpretation targets:\n")
        if infected_paras and uninfected_paras:
            fh.write(
                f"- Infected vs Uninfected parasitemia: mean {np.mean(infected_paras):.3f}% vs {np.mean(uninfected_paras):.3f}% | "
                f"median {np.median(infected_paras):.3f}% vs {np.median(uninfected_paras):.3f}% | "
                f"Cohen's d={cohen_d(infected_paras, uninfected_paras):.3f}\n"
            )
        if thick_paras and thin_paras:
            fh.write(
                f"- Thick vs Thin parasitemia: mean {np.mean(thick_paras):.3f}% vs {np.mean(thin_paras):.3f}% | "
                f"median {np.median(thick_paras):.3f}% vs {np.median(thin_paras):.3f}% | "
                f"Cohen's d={cohen_d(thick_paras, thin_paras):.3f}\n"
            )
        fh.write("\nGroup-level breakdown:\n")
        for r in group_stats:
            notes = []
            if float(r["mean_total_cells"]) < 5:
                notes.append("low detected cells")
            if float(r["pct_zero_cells"]) > 50.0:
                notes.append("many zero-cell images")
            if float(r["mean_avg_mnet"]) < 0.3:
                notes.append("weak classifier confidence")
            if float(r["pct_any_parasites"]) < 20.0 and r["group"].endswith("infected"):
                notes.append("infected group rarely crosses threshold")
            fh.write(f"- {r['group']}: mean parasitemia={r['mean_parasitemia']:.3f}%, median={r['median_parasitemia']:.3f}%, mean cells={r['mean_total_cells']:.2f}, mean mnet={r['mean_avg_mnet']:.3f}, notes={'; '.join(notes) if notes else 'ok'}\n")
        fh.write("\nSmear-type breakdown:\n")
        for r in smear_stats:
            fh.write(f"- {r['group']}: {r}\n")
        fh.write("\nLabel breakdown:\n")
        for r in label_stats:
            fh.write(f"- {r['group']}: {r}\n")

    # Save combined JSON for downstream inspection.
    summary_json = {
        "overall": overall_stats,
        "groups": group_stats,
        "labels": label_stats,
        "smears": smear_stats,
        "effect_sizes": {
            "infected_vs_uninfected_parasitemia_d": cohen_d(infected_paras, uninfected_paras),
            "thick_vs_thin_parasitemia_d": cohen_d(thick_paras, thin_paras),
        },
    }
    with (out_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)


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

    logging.info("Loading model: %s", args.model_path)
    model, resolved_model_input_size = load_diagnostic_model(
        args.model_path,
        fallback_input_size=(args.model_input_size, args.model_input_size),
    )

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

    if "temp_paths" not in locals():
        temp_paths: list[str] = []

    try:
        logging.info("Discovered %d images to process", len(samples))

        overlay_dir = args.output_dir / "overlays" if args.overlay_thick else None

        df = process_all(
            samples=samples,
            model=model,
            threshold=args.threshold,
            roi_size=args.roi_size,
            model_input_size=resolved_model_input_size,
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
