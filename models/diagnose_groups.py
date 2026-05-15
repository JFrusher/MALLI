"""Diagnostic tool to check group distribution in the dataset."""

from pathlib import Path
from collections import Counter
import argparse
from .roi_grabber_review import crawl_zip_samples, crawl_folder_samples


def diagnose_groups() -> None:
    parser = argparse.ArgumentParser(description="Diagnose group distribution in dataset")
    parser.add_argument("--zip-path", type=Path, default=Path("dataverse_files.zip"), help="Path to Dataverse ZIP")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to extracted dataset (alternative to ZIP)")
    args = parser.parse_args()

    # Crawl samples
    if args.dataset_root and args.dataset_root.exists():
        print(f"Crawling folder: {args.dataset_root}")
        samples = crawl_folder_samples(args.dataset_root, thick_min_dim=512)
    else:
        print(f"Crawling ZIP: {args.zip_path}")
        if not args.zip_path.exists():
            print(f"ERROR: ZIP not found at {args.zip_path}")
            return
        samples = crawl_zip_samples(args.zip_path, thick_min_dim=512)

    if not samples:
        print("ERROR: No samples found!")
        return

    print(f"\nTotal samples discovered: {len(samples)}\n")

    # Group statistics
    group_counter = Counter(s.group_key for s in samples)
    coverage_counter = Counter(s.coverage_key for s in samples)
    label_counter = Counter(s.label_name for s in samples)
    smear_counter = Counter(s.smear_type for s in samples)

    print("=" * 70)
    print("GROUP DISTRIBUTION (smear_label):")
    print("=" * 70)
    for group, count in sorted(group_counter.items()):
        print(f"  {group:30s} | {count:4d} samples")

    print("\n" + "=" * 70)
    print("COVERAGE DISTRIBUTION (smear_label_coverage):")
    print("=" * 70)
    for coverage, count in sorted(coverage_counter.items()):
        print(f"  {coverage:30s} | {count:4d} samples")

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION:")
    print("=" * 70)
    for label, count in sorted(label_counter.items()):
        print(f"  {label:30s} | {count:4d} samples")

    print("\n" + "=" * 70)
    print("SMEAR TYPE DISTRIBUTION:")
    print("=" * 70)
    for smear, count in sorted(smear_counter.items()):
        print(f"  {smear:30s} | {count:4d} samples")

    # Sample paths by group
    print("\n" + "=" * 70)
    print("SAMPLE PATHS (first 3 per group):")
    print("=" * 70)
    by_group = {}
    for sample in samples:
        if sample.group_key not in by_group:
            by_group[sample.group_key] = []
        by_group[sample.group_key].append(sample.source_id)

    for group in sorted(by_group.keys()):
        paths = by_group[group][:3]
        print(f"\n{group}:")
        for path in paths:
            print(f"  - {path}")
        if len(by_group[group]) > 3:
            print(f"  ... and {len(by_group[group]) - 3} more")

    print("\n" + "=" * 70)
    print("EXPECTED GROUPS FOR STRATIFIED SAMPLING:")
    print("=" * 70)
    print("The reviewer will cycle through these groups:")
    for i, group in enumerate(sorted(group_counter.keys()), 1):
        print(f"  {i}. {group}")


if __name__ == "__main__":
    diagnose_groups()
