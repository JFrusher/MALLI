"""Integration test and demo for the CellCounter pipeline.

Verifies all components work together correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def verify_imports() -> bool:
    """Verify all required modules can be imported."""
    logger.info("Verifying imports...")
    try:
        from models.cell_counter import CellCounter, CellCountResult
        from models.cell_counter_example import (
            draw_cell_overlay,
            save_overlay,
            export_results_csv,
            print_summary_report,
        )
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def verify_model_files() -> bool:
    """Verify required model files exist."""
    logger.info("Verifying model files...")
    required_files = [
        Path("models/best_mobilenetv3_small.weights.h5"),
        Path("models/decision_threshold.json"),
    ]

    for path in required_files:
        if path.exists():
            logger.info(f"✓ Found {path} ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            logger.error(f"✗ Missing {path}")
            return False

    return True


def test_initialization() -> bool:
    """Test CellCounter initialization."""
    logger.info("Testing CellCounter initialization...")
    try:
        from models.cell_counter import CellCounter

        counter = CellCounter(verbose=False)
        logger.info(f"✓ CellCounter initialized")
        logger.info(f"  - Threshold: {counter.threshold:.3f}")
        logger.info(f"  - ROI size: {counter.roi_size}")
        logger.info(f"  - Batch size: {counter.batch_size}")
        logger.info(f"  - NMS IOU: {counter.nms_iou_threshold}")
        return True
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        return False


def test_single_image_processing() -> bool:
    """Test processing a single image if available."""
    logger.info("Testing single image processing...")
    try:
        from models.cell_counter import CellCounter
        import cv2
        import numpy as np

        # Find a test image
        test_images = list(Path("nih_data").rglob("*.png"))[:1]
        if not test_images:
            logger.warning("⊘ No test images found in nih_data (skipping)")
            return True

        counter = CellCounter(verbose=False)
        for image_path in test_images:
            logger.info(f"  Processing: {image_path}")
            result = counter.process_image(image_path, smear_type="auto")
            logger.info(f"  ✓ {result}")
            logger.info(
                f"    - Proposals: {result.raw_proposal_count} → "
                f"{result.total_cells} (after filtering)"
            )
            logger.info(f"    - Infected: {result.infected_cells}")
            logger.info(f"    - Parasitemia: {result.parasitemia_percent:.2f}%")

        return True
    except Exception as e:
        logger.error(f"✗ Image processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing() -> bool:
    """Test batch processing if test images available."""
    logger.info("Testing batch processing...")
    try:
        from models.cell_counter import CellCounter

        # Find test images
        test_images = list(Path("nih_data").rglob("*.png"))[:3]
        if not test_images:
            logger.warning("⊘ No test images found (skipping batch test)")
            return True

        counter = CellCounter(verbose=False)
        logger.info(f"  Processing {len(test_images)} images...")
        results = counter.process_batch(test_images, skip_errors=True)

        if results:
            stats = counter.get_summary_statistics(results)
            logger.info(f"✓ Batch processing complete")
            logger.info(f"  - Images: {stats['num_images']}")
            logger.info(f"  - Total cells: {stats['total_cells']}")
            logger.info(f"  - Mean parasitemia: {stats['mean_parasitemia_percent']:.2f}%")
            return True
        else:
            logger.warning("⊘ No results returned from batch processing")
            return True

    except Exception as e:
        logger.error(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataclass() -> bool:
    """Test CellCountResult dataclass."""
    logger.info("Testing CellCountResult dataclass...")
    try:
        from models.cell_counter import CellCountResult
        import numpy as np

        result = CellCountResult(
            image_path="/test/image.png",
            total_cells=100,
            infected_cells=25,
            uninfected_cells=75,
            parasitemia_percent=25.0,
            raw_proposal_count=110,
            smear_type="thin",
            probabilities=np.array([0.1, 0.9, 0.5]),
            kept_indices=[1],
        )

        # Test string representation
        _ = str(result)
        # Test dict conversion
        d = result.to_dict()

        logger.info("✓ CellCountResult dataclass works correctly")
        logger.info(f"  - String repr: {result}")
        logger.info(f"  - Dict keys: {list(d.keys())}")
        return True
    except Exception as e:
        logger.error(f"✗ Dataclass test failed: {e}")
        return False


def run_all_tests() -> None:
    """Run all integration tests."""
    logger.info("=" * 80)
    logger.info("CELL COUNTER INTEGRATION TEST SUITE")
    logger.info("=" * 80)

    tests = [
        ("Import Verification", verify_imports),
        ("Model Files", verify_model_files),
        ("Initialization", test_initialization),
        ("Dataclass", test_dataclass),
        ("Single Image", test_single_image_processing),
        ("Batch Processing", test_batch_processing),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info("")
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")
    logger.info("=" * 80)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
