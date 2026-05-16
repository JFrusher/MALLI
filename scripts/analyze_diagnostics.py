#!/usr/bin/env python3
"""Analyze diagnostics CSV: filter zero-count images, remove outliers, and save comparative histograms.

Usage examples:
  python scripts/analyze_diagnostics.py \
    --input outputs/diagnostics/diagnostics_per_image.csv \
    --outdir outputs/diagnostics/analysis_plots

The script auto-detects a `count`-style column (e.g. `cell_count`, `count`, `n_cells`).
If none found, it will pick the first numeric column.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMERIC_HINTS = ("count", "cells", "n_cells", "num_cells", "cell_count", "parasite", "area", "mnet", "median", "std", "parasitemia")


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def infer_count_column(fieldnames):
    candidates = [name for name in fieldnames if re.search(r"\b(count|cells|n_cells|num_cells|cell_count)\b", name, re.I)]
    if candidates:
        return candidates[0]
    for name in fieldnames:
        if any(hint in name.lower() for hint in NUMERIC_HINTS):
            return name
    return None


def read_rows(path: str):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def numeric_columns(fieldnames, rows):
    numeric = []
    for name in fieldnames:
        values = [to_float(row.get(name)) for row in rows]
        non_empty = [v for v in values if v is not None]
        if non_empty and len(non_empty) >= max(3, len(values) // 10):
            numeric.append(name)
    return numeric


def column_values(rows, column):
    values = []
    for row in rows:
        value = to_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def mean(values):
    return sum(values) / len(values) if values else None


def stddev(values):
    if len(values) < 2:
        return None
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def remove_outliers_iqr(rows, columns, k=1.5):
    filtered = list(rows)
    for column in columns:
        values = sorted(column_values(filtered, column))
        if len(values) < 4:
            continue
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        filtered = [row for row in filtered if (to_float(row.get(column)) is None) or (low <= to_float(row.get(column)) <= high)]
    return filtered


def remove_outliers_zscore(rows, columns, thresh=3.0):
    filtered = list(rows)
    for column in columns:
        values = column_values(filtered, column)
        if len(values) < 2:
            continue
        m = mean(values)
        sd = stddev(values)
        if sd in (None, 0):
            continue
        filtered = [row for row in filtered if (to_float(row.get(column)) is None) or (abs((to_float(row.get(column)) - m) / sd) <= thresh)]
    return filtered


def histogram_bins(values, max_bins=40):
    values = [v for v in values if v is not None]
    if not values:
        return 10
    unique = len(set(values))
    return max(10, min(max_bins, int(math.sqrt(unique)) if unique > 1 else 10))


def plot_comparison(original, no_zero, cleaned, column, outdir):
    orig_vals = column_values(original, column)
    nz_vals = column_values(no_zero, column)
    clean_vals = column_values(cleaned, column)
    if not orig_vals:
        return None

    combined = [v for v in orig_vals + nz_vals + clean_vals if v is not None]
    bins = histogram_bins(combined)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    panels = [
        (axes[0], orig_vals, "Original", "C0"),
        (axes[1], nz_vals, "No zero-count images", "C1"),
        (axes[2], clean_vals, "No zeros + outliers removed", "C2"),
    ]

    for ax, values, title, color in panels:
        ax.hist(values, bins=bins, color=color, alpha=0.8, edgecolor="white", density=True)
        if len(values) >= 2:
            ax.axvline(mean(values), color="black", linestyle="--", linewidth=1, label="mean")
            ax.axvline(percentile(sorted(values), 0.5), color="black", linestyle=":", linewidth=1, label="median")
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(column, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", column)
    path = os.path.join(outdir, f"{base}_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
    ax2.hist(orig_vals, bins=bins, density=True, alpha=0.35, color="C0", label=f"original (n={len(orig_vals)})")
    ax2.hist(nz_vals, bins=bins, density=True, alpha=0.35, color="C1", label=f"no zeros (n={len(nz_vals)})")
    ax2.hist(clean_vals, bins=bins, density=True, alpha=0.35, color="C2", label=f"cleaned (n={len(clean_vals)})")
    if len(orig_vals) >= 2:
        ax2.axvline(mean(orig_vals), color="C0", linestyle="--", linewidth=1)
    if len(nz_vals) >= 2:
        ax2.axvline(mean(nz_vals), color="C1", linestyle="--", linewidth=1)
    if len(clean_vals) >= 2:
        ax2.axvline(mean(clean_vals), color="C2", linestyle="--", linewidth=1)
    ax2.set_title(f"Overlay comparison: {column}")
    ax2.set_xlabel(column)
    ax2.set_ylabel("Density")
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8)
    overlay_path = os.path.join(outdir, f"{base}_overlay.png")
    fig2.tight_layout()
    fig2.savefig(overlay_path, dpi=160)
    plt.close(fig2)

    return path


def write_cleaned_csv(rows, fieldnames, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze diagnostics CSV and save comparative histograms")
    parser.add_argument("--input", "-i", required=True, help="Input CSV, such as outputs/diagnostics/diagnostics_per_image.csv")
    parser.add_argument("--outdir", "-o", default="outputs/diagnostics/analysis_plots", help="Output directory for plots")
    parser.add_argument("--count-col", "-c", default=None, help="Column name that represents per-image cell count")
    parser.add_argument("--outlier-method", choices=["iqr", "zscore", "none"], default="iqr")
    parser.add_argument("--iqr-k", type=float, default=1.5)
    parser.add_argument("--z-thresh", type=float, default=3.0)
    parser.add_argument("--no-pairplot", action="store_true", help="Skip extra multi-metric scatter overview")
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    fieldnames, rows = read_rows(args.input)
    if not rows or not fieldnames:
        print("CSV is empty or missing headers. Exiting.")
        return

    count_col = args.count_col or infer_count_column(fieldnames)
    if not count_col:
        print("Could not infer a count column. Use --count-col to set one explicitly.")
        return

    numeric_cols = numeric_columns(fieldnames, rows)
    if count_col not in numeric_cols:
        numeric_cols.insert(0, count_col)

    original_rows = rows
    zero_filtered_rows = [row for row in original_rows if to_float(row.get(count_col)) not in (0, 0.0)]

    if args.outlier_method == "iqr":
        cleaned_rows = remove_outliers_iqr(zero_filtered_rows, numeric_cols, k=args.iqr_k)
    elif args.outlier_method == "zscore":
        cleaned_rows = remove_outliers_zscore(zero_filtered_rows, numeric_cols, thresh=args.z_thresh)
    else:
        cleaned_rows = list(zero_filtered_rows)

    print(f"Detected count column: {count_col}")
    print(f"Rows before: {len(original_rows)}")
    print(f"Rows after removing zero-count images: {len(zero_filtered_rows)}")
    print(f"Rows after outlier removal ({args.outlier_method}): {len(cleaned_rows)}")

    summary_path = os.path.join(args.outdir, "cleaned_diagnostics_per_image.csv")
    write_cleaned_csv(cleaned_rows, fieldnames, summary_path)

    saved = []
    for column in numeric_cols:
        path = plot_comparison(original_rows, zero_filtered_rows, cleaned_rows, column, args.outdir)
        if path:
            saved.append(path)

    if not args.no_pairplot and len(numeric_cols) >= 2:
        try:
            scatter_cols = [col for col in numeric_cols if len(column_values(cleaned_rows, col)) >= 2]
            scatter_cols = scatter_cols[:4]
            if len(scatter_cols) >= 2:
                fig, axes = plt.subplots(len(scatter_cols) - 1, len(scatter_cols) - 1, figsize=(10, 10))
                if len(scatter_cols) - 1 == 1:
                    axes = [[axes]]
                for i in range(1, len(scatter_cols)):
                    for j in range(i):
                        ax = axes[i - 1][j]
                        x = column_values(cleaned_rows, scatter_cols[j])
                        y = column_values(cleaned_rows, scatter_cols[i])
                        n = min(len(x), len(y))
                        ax.scatter(x[:n], y[:n], s=10, alpha=0.35)
                        ax.set_xlabel(scatter_cols[j])
                        ax.set_ylabel(scatter_cols[i])
                        ax.grid(alpha=0.2)
                fig.suptitle("Cleaned metric relationships", fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                pair_path = os.path.join(args.outdir, "cleaned_metric_scatter.png")
                fig.savefig(pair_path, dpi=160)
                plt.close(fig)
                saved.append(pair_path)
        except Exception as exc:
            print(f"Scatter overview failed: {exc}")

    print(f"Saved cleaned CSV to {summary_path}")
    print(f"Saved {len(saved)} plot files to {args.outdir}")
    for path in saved:
        print(f" - {path}")


if __name__ == '__main__':
    main()
