"""Example usage and CLI for the CellCounter pipeline.

Demonstrates:
1. Single image processing
2. Batch processing
3. Directory scanning
4. Statistics and reporting
5. CSV export
6. Visualization with overlays
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from models.cell_counter import CellCounter, CellCountResult


logger = logging.getLogger(__name__)


def draw_cell_overlay(
    image_bgr: np.ndarray,
    result: CellCountResult,
    color_infected: tuple[int, int, int] = (0, 0, 255),
    color_uninfected: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """Draw bounding boxes on image with infection status.

    Args:
        image_bgr: Original image in BGR format
        result: CellCountResult with proposal and prediction data
        color_infected: BGR color for infected cells (red by default)
        color_uninfected: BGR color for uninfected cells (green by default)
        thickness: Box line thickness
        font_scale: Text scale for labels

    Returns:
        Image with overlay boxes and labels
    """
    overlay = image_bgr.copy()

    for idx, proposal in enumerate(result.proposals):
        x1, y1, x2, y2 = proposal.box
        prob = result.probabilities[idx]

        is_infected = idx in result.kept_indices
        color = color_infected if is_infected else color_uninfected
        label_prefix = "INF" if is_infected else "HLT"

        # Draw rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label_text = f"{label_prefix} {prob:.2f}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        text_x = x1
        text_y = max(15, y1 - 5)

        cv2.putText(
            overlay,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    # Add summary text
    summary_text = f"Total: {result.total_cells} | Infected: {result.infected_cells} | {result.parasitemia_percent:.1f}%"
    cv2.putText(
        overlay,
        summary_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


def save_overlay(
    image_bgr: np.ndarray,
    result: CellCountResult,
    output_path: Path,
) -> None:
    """Save image with overlay to disk.

    Args:
        image_bgr: Original image
        result: Processing result with proposals
        output_path: Where to save the overlay image
    """
    overlay = draw_cell_overlay(image_bgr, result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    logger.info(f"Saved overlay: {output_path}")


def export_results_csv(
    results: list[CellCountResult],
    output_path: Path,
) -> None:
    """Export results to CSV file.

    Args:
        results: List of CellCountResult objects
        output_path: Path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_path",
            "smear_type",
            "total_cells",
            "infected_cells",
            "uninfected_cells",
            "parasitemia_percent",
            "raw_proposal_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.to_dict())

    logger.info(f"Exported {len(results)} results to {output_path}")


def print_summary_report(
    results: list[CellCountResult],
    counter: CellCounter,
) -> None:
    """Print human-readable summary report.

    Args:
        results: List of results
        counter: CellCounter instance (for settings reference)
    """
    if not results:
        print("No results to report")
        return

    stats = counter.get_summary_statistics(results)

    print("\n" + "=" * 80)
    print("CELL COUNTING SUMMARY REPORT")
    print("=" * 80)
    print(f"Images processed: {stats['num_images']}")
    print(f"Total cells detected: {stats['total_cells']}")
    print(f"  - Infected: {stats['total_infected']}")
    print(f"  - Uninfected: {stats['total_uninfected']}")
    print(f"\nParasitemia Statistics:")
    print(f"  - Mean: {stats['mean_parasitemia_percent']:.2f}%")
    print(f"  - Std Dev: {stats['std_parasitemia_percent']:.2f}%")
    print(f"  - Min: {stats['min_parasitemia_percent']:.2f}%")
    print(f"  - Max: {stats['max_parasitemia_percent']:.2f}%")
    print(f"\nSmear Type Distribution:")
    for smear_type, count in stats["smear_type_distribution"].items():
        print(f"  - {smear_type}: {count}")
    print(f"\nSettings:")
    print(f"  - Threshold: {counter.threshold:.3f}")
    print(f"  - ROI Size: {counter.roi_size}")
    print(f"  - NMS IOU Threshold: {counter.nms_iou_threshold}")
    print("=" * 80 + "\n")


def main() -> None:
    """Command-line interface for cell counting."""
    parser = argparse.ArgumentParser(
        description="Cell counting pipeline using ROI extraction + MobileNetV3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python -m models.cell_counter_example --image path/to/image.png

  # Process all PNGs in directory
  python -m models.cell_counter_example --directory nih_data --pattern "*.png" --recursive

  # With overlays and CSV export
  python -m models.cell_counter_example \\
    --directory nih_data \\
    --output-csv results.csv \\
    --output-overlays overlays/ \\
    --recursive

  # Specific smear type (skip auto-detection)
  python -m models.cell_counter_example \\
    --image image.png \\
    --smear-type thick
        """,
    )

    parser.add_argument("--image", type=Path, help="Path to single image file")
    parser.add_argument("--directory", type=Path, help="Path to directory with images")
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern for images (default: *.png)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories",
    )
    parser.add_argument(
        "--smear-type",
        choices=["thin", "thick", "auto"],
        default="auto",
        help="Cell smear type (default: auto-detect)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default="models/best_mobilenetv3_small.weights.h5",
        help="Path to model weights",
    )
    parser.add_argument(
        "--threshold",
        type=Path,
        default="models/decision_threshold.json",
        help="Path to decision threshold JSON",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=128,
        help="ROI crop size (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Model batch size (default: 32)",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IOU threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--output-overlays",
        type=Path,
        help="Directory to save overlay images",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.directory:
        parser.error("Must specify either --image or --directory")

    if args.image and args.directory:
        parser.error("Cannot specify both --image and --directory")

    # Initialize logger
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Initialize counter
    logger.info("Initializing CellCounter...")
    counter = CellCounter(
        weights_path=args.weights,
        threshold_path=args.threshold,
        roi_size=args.roi_size,
        batch_size=args.batch_size,
        nms_iou_threshold=args.nms_iou,
        verbose=args.verbose,
    )

    # Process images
    results: list[CellCountResult] = []

    if args.image:
        # Single image
        logger.info(f"Processing single image: {args.image}")
        try:
            result = counter.process_image(args.image, smear_type=args.smear_type)
            results.append(result)
            print(f"\n{result}\n")

            if args.output_overlays:
                image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
                if image_bgr is not None:
                    output_path = (
                        Path(args.output_overlays)
                        / f"{args.image.stem}_overlay{args.image.suffix}"
                    )
                    save_overlay(image_bgr, result, output_path)

        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return

    else:
        # Directory
        logger.info(f"Processing directory: {args.directory}")
        results = counter.process_directory(
            args.directory,
            pattern=args.pattern,
            recursive=args.recursive,
            smear_type=args.smear_type,
            skip_errors=True,
        )

        if args.output_overlays:
            logger.info("Generating overlay images...")
            for result in results:
                try:
                    image_path = Path(result.image_path)
                    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image_bgr is None:
                        continue

                    output_path = (
                        Path(args.output_overlays)
                        / image_path.parent.name
                        / f"{image_path.stem}_overlay{image_path.suffix}"
                    )
                    save_overlay(image_bgr, result, output_path)
                except Exception as e:
                    logger.error(f"Failed to save overlay for {result.image_path}: {e}")

    # Export results
    if args.output_csv and results:
        export_results_csv(results, args.output_csv)

    # Print report
    print_summary_report(results, counter)


if __name__ == "__main__":
    main()
