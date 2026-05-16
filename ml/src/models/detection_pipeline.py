"""Python implementation of the mobile Dart detection pipeline.

Provides SAHI-style tiling inference, green channel preprocessing,
and post-inference filtering (geometric + NMS) for blood smear analysis.

This module mirrors the Dart implementation in lib/services/detection_pipeline.dart
for easy comparison and validation between mobile and server pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Callable
import numpy as np
import cv2


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box detection with confidence score."""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int = 0

    @property
    def area(self) -> float:
        """Calculate area of bounding box."""
        return self.width * self.height

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Get box as (x1, y1, x2, y2) format."""
        return (
            int(self.x),
            int(self.y),
            int(self.x + self.width),
            int(self.y + self.height),
        )

    @property
    def corners(self) -> list[float]:
        """Get box corners as [x1, y1, x2, y2]."""
        return [self.x, self.y, self.x + self.width, self.y + self.height]

    def copy_with(
        self,
        x: float | None = None,
        y: float | None = None,
        width: float | None = None,
        height: float | None = None,
        confidence: float | None = None,
        class_id: int | None = None,
    ) -> BoundingBox:
        """Create a copy with modified fields."""
        return BoundingBox(
            x=x if x is not None else self.x,
            y=y if y is not None else self.y,
            width=width if width is not None else self.width,
            height=height if height is not None else self.height,
            confidence=confidence if confidence is not None else self.confidence,
            class_id=class_id if class_id is not None else self.class_id,
        )

    def __str__(self) -> str:
        return f"BBox(x:{self.x:.1f}, y:{self.y:.1f}, w:{self.width:.1f}, h:{self.height:.1f}, conf:{self.confidence:.3f})"


class ImagePreProcessor:
    """Preprocesses images by emphasizing green channel for parasite detection."""

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
        """Boost green channel and suppress red/blue for parasite detection.

        Args:
            image_bgr: Input BGR image (H x W x 3), uint8

        Returns:
            Preprocessed BGR image with green emphasis
        """
        image_float = image_bgr.astype(np.float32)

        # Extract channels
        b = image_float[:, :, 0]
        g = image_float[:, :, 1]
        r = image_float[:, :, 2]

        # Boost green (+30%), suppress red/blue (-30%)
        g = np.clip(g * 1.3, 0, 255)
        r = np.clip(r * 0.7, 0, 255)
        b = np.clip(b * 0.7, 0, 255)

        # Recombine and convert back to uint8
        result = np.stack([b, g, r], axis=2).astype(np.uint8)
        return result

    @staticmethod
    def apply_gaussian_blur(image_bgr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        kernel_size = int(2 * np.ceil(3 * sigma)) + 1
        return cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), sigma)

    @staticmethod
    def enhance_contrast(image_bgr: np.ndarray) -> np.ndarray:
        """Enhance contrast via histogram stretching."""
        result = np.zeros_like(image_bgr, dtype=np.uint8)

        for ch in range(3):
            channel = image_bgr[:, :, ch]
            min_val = channel.min()
            max_val = channel.max()
            range_val = max(1, max_val - min_val)

            stretched = ((channel - min_val) / range_val * 255).astype(np.uint8)
            result[:, :, ch] = stretched

        return result


class TilingInferenceEngine:
    """SAHI-style overlapping tile processing for high-resolution images."""

    def __init__(self, tile_size: int = 640, overlap_ratio: float = 0.2):
        """Initialize tiling engine.

        Args:
            tile_size: Size of each tile in pixels (default 640)
            overlap_ratio: Overlap between adjacent tiles (0-1, default 0.2 = 20%)
        """
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio

    def generate_tiles(self, image_height: int, image_width: int) -> list[dict[str, int]]:
        """Generate overlapping tile coordinates.

        Args:
            image_height: Height of full image
            image_width: Width of full image

        Returns:
            List of {'x', 'y', 'width', 'height'} dicts for each tile
        """
        tiles = []
        stride = int(self.tile_size * (1 - self.overlap_ratio))

        y = 0
        while y < image_height:
            x = 0
            while x < image_width:
                tile_w = min(self.tile_size, image_width - x)
                tile_h = min(self.tile_size, image_height - y)

                tiles.append({
                    'x': x,
                    'y': y,
                    'width': tile_w,
                    'height': tile_h,
                })

                x += stride
                if x + self.tile_size >= image_width:
                    break

            y += stride
            if y + self.tile_size >= image_height:
                break

        return tiles

    @staticmethod
    def extract_tile(
        full_image: np.ndarray,
        tile_x: int,
        tile_y: int,
        tile_width: int,
        tile_height: int,
        target_size: int,
    ) -> np.ndarray:
        """Extract a tile from full image, padding if necessary.

        Args:
            full_image: Full image (H x W x 3), BGR uint8
            tile_x: Top-left x coordinate
            tile_y: Top-left y coordinate
            tile_width: Tile width
            tile_height: Tile height
            target_size: Output tile size (typically same as tile_width/height)

        Returns:
            Extracted and padded tile (target_size x target_size x 3)
        """
        tile = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        source_x2 = min(full_image.shape[1], tile_x + target_size)
        source_y2 = min(full_image.shape[0], tile_y + target_size)

        if source_x2 <= tile_x or source_y2 <= tile_y:
            return tile

        crop = full_image[tile_y:source_y2, tile_x:source_x2]
        tile[: crop.shape[0], : crop.shape[1]] = crop
        return tile

    def process_tiles(
        self,
        image: np.ndarray,
        inference_callback: Callable[[np.ndarray], list[BoundingBox]],
    ) -> list[BoundingBox]:
        """Process all tiles and collect detections.

        Args:
            image: Full image (H x W x 3), BGR uint8
            inference_callback: Async function that runs inference on a tile
                               and returns list of BoundingBox in tile coordinates

        Returns:
            Combined list of detections mapped to full image coordinates
        """
        all_detections = []
        tiles = self.generate_tiles(image.shape[0], image.shape[1])

        for tile_info in tiles:
            tile_x = tile_info['x']
            tile_y = tile_info['y']
            tile_width = tile_info['width']
            tile_height = tile_info['height']

            # Extract tile
            tile_image = self.extract_tile(
                image,
                tile_x,
                tile_y,
                tile_width,
                tile_height,
                self.tile_size,
            )

            # Run inference on tile
            tile_detections = inference_callback(tile_image)

            # Map tile-local coordinates to full image coordinates
            for detection in tile_detections:
                mapped = detection.copy_with(
                    x=detection.x + tile_x,
                    y=detection.y + tile_y,
                )
                all_detections.append(mapped)

        return all_detections


class PostInferenceFilter:
    """Geometric filtering and Non-Maximum Suppression for detections."""

    def __init__(
        self,
        min_area: float = 100,
        max_area_ratio: float = 0.15,
        nms_threshold: float = 0.5,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.0,
        tile_size: int = 640,
    ):
        """Initialize post-inference filter.

        Args:
            min_area: Minimum bounding box area (pixels)
            max_area_ratio: Max area as fraction of tile area (0.15 = 15%)
            nms_threshold: IoU threshold for NMS (0.5 typical)
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            tile_size: Size of inference tiles (for area ratio calculation)
        """
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.nms_threshold = nms_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.tile_size = tile_size

    def geometric_filter(self, detections: Sequence[BoundingBox]) -> list[BoundingBox]:
        """Apply geometric constraints to filter invalid detections.

        Args:
            detections: List of bounding boxes

        Returns:
            Filtered list keeping only valid boxes
        """
        max_area = (self.tile_size * self.tile_size) * self.max_area_ratio
        filtered = []

        for box in detections:
            area = box.area
            aspect_ratio = box.width / max(1, box.height)

            is_valid_area = area >= self.min_area and area <= max_area
            is_valid_aspect = (aspect_ratio >= self.min_aspect_ratio and
                              aspect_ratio <= self.max_aspect_ratio)

            if is_valid_area and is_valid_aspect:
                filtered.append(box)

        return filtered

    @staticmethod
    def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two boxes.

        Args:
            box1: First bounding box
            box2: Second bounding box

        Returns:
            IoU value between 0 and 1
        """
        c1 = box1.corners
        c2 = box2.corners

        # Calculate intersection
        intersect_x1 = max(c1[0], c2[0])
        intersect_y1 = max(c1[1], c2[1])
        intersect_x2 = min(c1[2], c2[2])
        intersect_y2 = min(c1[3], c2[3])

        if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
            return 0.0

        intersection_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

        # Calculate union
        box1_area = box1.area
        box2_area = box2.area
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def apply_nms(self, detections: Sequence[BoundingBox]) -> list[BoundingBox]:
        """Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections: List of bounding boxes

        Returns:
            NMS-filtered detections, sorted by confidence descending
        """
        if not detections:
            return []

        # Sort by confidence descending
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        kept = []
        suppressed = set()

        for i, current in enumerate(sorted_detections):
            if i in suppressed:
                continue

            kept.append(current)

            # Compare with remaining boxes
            for j in range(i + 1, len(sorted_detections)):
                if j in suppressed:
                    continue

                iou = self.calculate_iou(current, sorted_detections[j])

                if iou > self.nms_threshold:
                    suppressed.add(j)

        return kept

    async def filter_and_deduplicate(
        self,
        raw_detections: Sequence[BoundingBox],
    ) -> list[BoundingBox]:
        """Complete post-processing pipeline: geometric filter + NMS.

        Args:
            raw_detections: List of raw detections from inference

        Returns:
            Filtered and deduplicated detections
        """
        # Step 1: Apply geometric constraints
        geometry_filtered = self.geometric_filter(raw_detections)

        # Step 2: Apply NMS
        final_filtered = self.apply_nms(geometry_filtered)

        return final_filtered


class BloodSmearDetectionPipeline:
    """Orchestrator for complete blood smear detection pipeline."""

    def __init__(
        self,
        tiler: TilingInferenceEngine | None = None,
        filter: PostInferenceFilter | None = None,
    ):
        """Initialize pipeline.

        Args:
            tiler: TilingInferenceEngine instance
            filter: PostInferenceFilter instance
        """
        self.preprocessor = ImagePreProcessor()
        self.tiler = tiler or TilingInferenceEngine()
        self.filter = filter or PostInferenceFilter()

    def detect_cells(
        self,
        image: np.ndarray,
        inference_callback: Callable[[np.ndarray], list[BoundingBox]],
    ) -> list[BoundingBox]:
        """Complete end-to-end detection pipeline.

        Args:
            image: Input BGR image (H x W x 3), uint8
            inference_callback: Function to run inference on tile

        Returns:
            Final detections after post-processing
        """
        # Step 1: Preprocess image
        preprocessed = self.preprocessor.preprocess_image(image)

        # Step 2: Process tiles and collect raw detections
        tile_detections = self.tiler.process_tiles(preprocessed, inference_callback)

        # Step 3: Post-processing (filter + NMS)
        final_detections = self.filter.apply_nms(
            self.filter.geometric_filter(tile_detections)
        )

        return final_detections
