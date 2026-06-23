import 'dart:math';
import 'package:image/image.dart' as img;

/// ============================================================================
/// Bounding Box Data Structure
/// ============================================================================
/// Represents a single detection with normalized coordinates and confidence.
class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;
  final double confidence;
  final int classId;

  BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.confidence,
    this.classId = 0,
  });

  /// Calculate area of bounding box
  double get area => width * height;

  /// Get box as [x1, y1, x2, y2] format
  List<double> get corners => [x, y, x + width, y + height];

  /// Create a copy with modified fields
  BoundingBox copyWith({
    double? x,
    double? y,
    double? width,
    double? height,
    double? confidence,
    int? classId,
  }) {
    return BoundingBox(
      x: x ?? this.x,
      y: y ?? this.y,
      width: width ?? this.width,
      height: height ?? this.height,
      confidence: confidence ?? this.confidence,
      classId: classId ?? this.classId,
    );
  }

  @override
  String toString() =>
      'BoundingBox(x:$x, y:$y, w:$width, h:$height, conf:${confidence.toStringAsFixed(3)})';
}

/// ============================================================================
/// IMAGE PRE-PROCESSOR
/// ============================================================================
/// Optimizes blood smear images by isolating and weighting the green channel.
/// Malaria parasites and RBC boundaries show highest contrast in green spectrum.
class ImagePreProcessor {
  /// Processes raw image data to emphasize green channel.
  ///
  /// Returns a new image with enhanced green channel contrast.
  /// Efficient for mobile by processing in-memory without disk I/O.
  static Future<img.Image> preprocessImage(img.Image sourceImage) async {
    final int width = sourceImage.width;
    final int height = sourceImage.height;

    // Create output image
    final result = img.Image(
      width: width,
      height: height,
      numChannels: 3,
    );

    // Process each pixel: enhance green channel, suppress R and B
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = sourceImage.getPixelRgba(x, y);

        // Extract RGB channels (0-255)
        int r = (pixel >> 16) & 0xFF;
        int g = (pixel >> 8) & 0xFF;
        int b = pixel & 0xFF;

        // Boost green channel (highest contrast for parasites/RBC)
        // Suppress red and blue to reduce background noise
        g = min(255, (g * 1.3).toInt()); // +30% boost
        r = max(0, (r * 0.7).toInt()); // -30% suppression
        b = max(0, (b * 0.7).toInt()); // -30% suppression

        // Ensure values stay in valid range
        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        // Set pixel in result image
        result.setPixelRgba(x, y, (r << 16) | (g << 8) | b | 0xFF000000);
      }
    }

    return result;
  }

  /// Applies Gaussian blur for noise reduction (optional preprocessing step)
  static Future<img.Image> applyGaussianBlur(
    img.Image image, {
    double sigma = 1.0,
  }) async {
    return img.gaussianBlur(image, radius: sigma.toInt());
  }

  /// Enhances contrast using CLAHE-like adaptive histogram equalization
  static Future<img.Image> enhanceContrast(img.Image image) async {
    final int width = image.width;
    final int height = image.height;

    final result = img.Image(
      width: width,
      height: height,
      numChannels: 3,
    );

    // Simple contrast enhancement: stretch histogram
    int minR = 255, maxR = 0;
    int minG = 255, maxG = 0;
    int minB = 255, maxB = 0;

    // Find min/max for each channel
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixelRgba(x, y);
        int r = (pixel >> 16) & 0xFF;
        int g = (pixel >> 8) & 0xFF;
        int b = pixel & 0xFF;

        minR = min(minR, r);
        maxR = max(maxR, r);
        minG = min(minG, g);
        maxG = max(maxG, g);
        minB = min(minB, b);
        maxB = max(maxB, b);
      }
    }

    // Normalize to full range
    final rangeR = max(1, maxR - minR);
    final rangeG = max(1, maxG - minG);
    final rangeB = max(1, maxB - minB);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixelRgba(x, y);
        int r = (pixel >> 16) & 0xFF;
        int g = (pixel >> 8) & 0xFF;
        int b = pixel & 0xFF;

        // Stretch to full range
        r = (((r - minR) / rangeR) * 255).toInt().clamp(0, 255);
        g = (((g - minG) / rangeG) * 255).toInt().clamp(0, 255);
        b = (((b - minB) / rangeB) * 255).toInt().clamp(0, 255);

        result.setPixelRgba(x, y, (r << 16) | (g << 8) | b | 0xFF000000);
      }
    }

    return result;
  }
}

/// ============================================================================
/// TILING INFERENCE ENGINE (SAHI-style)
/// ============================================================================
/// Processes high-resolution images by slicing into overlapping tiles.
/// Maps localized detections back to full image coordinates.
class TilingInferenceEngine {
  final int tileSize;
  final double overlapRatio;

  /// Creates a tiling engine with configurable tile size and overlap.
  ///
  /// [tileSize]: Expected model input size (e.g., 640x640)
  /// [overlapRatio]: Overlap percentage (0.0-1.0). Default 0.2 = 20%
  TilingInferenceEngine({
    this.tileSize = 640,
    this.overlapRatio = 0.2,
  });

  /// Generates overlapping tile coordinates for a given image.
  ///
  /// Returns list of [x, y, tileWidth, tileHeight] for each tile.
  List<Map<String, int>> generateTiles(int imageWidth, int imageHeight) {
    final tiles = <Map<String, int>>[];
    final stride = (tileSize * (1 - overlapRatio)).toInt();

    for (int y = 0; y < imageHeight; y += stride) {
      for (int x = 0; x < imageWidth; x += stride) {
        int tileW = min(tileSize, imageWidth - x);
        int tileH = min(tileSize, imageHeight - y);

        tiles.add({
          'x': x,
          'y': y,
          'width': tileW,
          'height': tileH,
        });

        // Stop if we've covered the full width
        if (x + tileSize >= imageWidth) break;
      }

      // Stop if we've covered the full height
      if (y + tileSize >= imageHeight) break;
    }

    return tiles;
  }

  /// Extracts a single tile from the full image.
  ///
  /// Padding is applied if tile extends beyond image boundaries.
  static Future<img.Image> extractTile(
    img.Image fullImage,
    int tileX,
    int tileY,
    int tileWidth,
    int tileHeight,
    int targetSize,
  ) async {
    final tile = img.Image(
      width: targetSize,
      height: targetSize,
      numChannels: 3,
    );

    // Copy pixels from full image, pad if necessary
    for (int py = 0; py < targetSize; py++) {
      for (int px = 0; px < targetSize; px++) {
        int sourceX = tileX + px;
        int sourceY = tileY + py;

        int pixel;
        if (sourceX >= 0 &&
            sourceX < fullImage.width &&
            sourceY >= 0 &&
            sourceY < fullImage.height) {
          pixel = fullImage.getPixelRgba(sourceX, sourceY);
        } else {
          // Pad with black (0,0,0)
          pixel = 0xFF000000;
        }

        tile.setPixelRgba(px, py, pixel);
      }
    }

    return tile;
  }

  /// Processes all tiles from an image and collects raw detections.
  ///
  /// [image]: Source image
  /// [inferenceCallback]: Async function that runs inference on a tile image
  ///                      and returns list of BoundingBoxes in tile coordinates
  /// Returns: Combined list of all detections mapped to full image coordinates
  Future<List<BoundingBox>> processTiles(
    img.Image image,
    Future<List<BoundingBox>> Function(img.Image tileImage) inferenceCallback,
  ) async {
    final allDetections = <BoundingBox>[];
    final tiles = generateTiles(image.width, image.height);

    // Process each tile sequentially to minimize memory footprint
    for (final tile in tiles) {
      final tileX = tile['x']!;
      final tileY = tile['y']!;
      final tileWidth = tile['width']!;
      final tileHeight = tile['height']!;

      // Extract tile
      final tileImage = await extractTile(
        image,
        tileX,
        tileY,
        tileWidth,
        tileHeight,
        tileSize,
      );

      // Run inference on tile
      final tileDetections = await inferenceCallback(tileImage);

      // Map tile-local coordinates to full image coordinates
      for (final detection in tileDetections) {
        final mapped = detection.copyWith(
          x: detection.x + tileX,
          y: detection.y + tileY,
        );
        allDetections.add(mapped);
      }
    }

    return allDetections;
  }
}

/// ============================================================================
/// POST-INFERENCE FILTER
/// ============================================================================
/// Applies geometric constraints and Non-Maximum Suppression (NMS) to detections.
class PostInferenceFilter {
  final double minArea;
  final double maxAreaRatio;
  final double nmsThreshold;
  final double minAspectRatio;
  final double maxAspectRatio;
  final int tileSize;

  /// Creates a post-inference filter with configurable thresholds.
  ///
  /// [minArea]: Minimum bounding box area (pixels) to keep
  /// [maxAreaRatio]: Max allowed area as fraction of tile area (0.15 = 15%)
  /// [nmsThreshold]: IoU threshold for NMS (0.5 typical)
  /// [minAspectRatio]: Minimum width/height ratio (0.3 typical for cells)
  /// [maxAspectRatio]: Maximum width/height ratio (3.0 typical for cells)
  /// [tileSize]: Size of inference tiles (for area ratio calculation)
  PostInferenceFilter({
    this.minArea = 100,
    this.maxAreaRatio = 0.15,
    this.nmsThreshold = 0.5,
    this.minAspectRatio = 0.3,
    this.maxAspectRatio = 3.0,
    this.tileSize = 640,
  });

  /// Applies geometric filtering to remove invalid detections.
  List<BoundingBox> geometricFilter(List<BoundingBox> detections) {
    final maxArea = (tileSize * tileSize) * maxAreaRatio;
    final filtered = <BoundingBox>[];

    for (final box in detections) {
      final area = box.area;
      final aspectRatio = box.width / max(1, box.height);

      // Filter conditions
      final isValidArea = area >= minArea && area <= maxArea;
      final isValidAspect =
          aspectRatio >= minAspectRatio && aspectRatio <= maxAspectRatio;

      if (isValidArea && isValidAspect) {
        filtered.add(box);
      }
    }

    return filtered;
  }

  /// Calculates Intersection over Union (IoU) between two boxes.
  static double calculateIoU(BoundingBox box1, BoundingBox box2) {
    // Get corners [x1, y1, x2, y2]
    final c1 = box1.corners;
    final c2 = box2.corners;

    // Calculate intersection
    final intersectX1 = max(c1[0], c2[0]);
    final intersectY1 = max(c1[1], c2[1]);
    final intersectX2 = min(c1[2], c2[2]);
    final intersectY2 = min(c1[3], c2[3]);

    // If no intersection, IoU = 0
    if (intersectX1 >= intersectX2 || intersectY1 >= intersectY2) {
      return 0.0;
    }

    final intersectionArea =
        (intersectX2 - intersectX1) * (intersectY2 - intersectY1);

    // Calculate union
    final box1Area = box1.area;
    final box2Area = box2.area;
    final unionArea = box1Area + box2Area - intersectionArea;

    return unionArea > 0 ? intersectionArea / unionArea : 0.0;
  }

  /// Applies Non-Maximum Suppression (NMS) to remove overlapping detections.
  ///
  /// Keeps higher-confidence boxes and removes lower-confidence overlaps.
  List<BoundingBox> applyNMS(List<BoundingBox> detections) {
    if (detections.isEmpty) return [];

    // Sort by confidence (descending)
    final sorted = List<BoundingBox>.from(detections)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final kept = <BoundingBox>[];
    final suppressed = <int>{};

    for (int i = 0; i < sorted.length; i++) {
      if (suppressed.contains(i)) continue;

      final current = sorted[i];
      kept.add(current);

      // Compare with all remaining boxes
      for (int j = i + 1; j < sorted.length; j++) {
        if (suppressed.contains(j)) continue;

        final iou = calculateIoU(current, sorted[j]);

        // Suppress if IoU exceeds threshold
        if (iou > nmsThreshold) {
          suppressed.add(j);
        }
      }
    }

    return kept;
  }

  /// Complete post-processing pipeline: geometric filter + NMS
  Future<List<BoundingBox>> filterAndDeduplicate(
    List<BoundingBox> rawDetections,
  ) async {
    // Step 1: Apply geometric constraints
    final geometryFiltered = geometricFilter(rawDetections);

    // Step 2: Apply NMS to remove overlaps
    final final_filtered = applyNMS(geometryFiltered);

    return final_filtered;
  }
}

/// ============================================================================
/// INTEGRATION HELPER
/// ============================================================================
/// Optional convenience class to orchestrate the complete pipeline.
class BloodSmearDetectionPipeline {
  final ImagePreProcessor preprocessor;
  final TilingInferenceEngine tiler;
  final PostInferenceFilter filter;

  BloodSmearDetectionPipeline({
    required this.tiler,
    required this.filter,
  }) : preprocessor = ImagePreProcessor();

  /// Complete end-to-end detection pipeline.
  ///
  /// 1. Preprocesses image (green channel emphasis)
  /// 2. Tiles the image with overlaps
  /// 3. Runs inference on each tile (via callback)
  /// 4. Maps detections to full coordinates
  /// 5. Applies post-processing filter
  Future<List<BoundingBox>> detectCells(
    img.Image image,
    Future<List<BoundingBox>> Function(img.Image tileImage)
        inferenceCallback,
  ) async {
    // Step 1: Preprocess image
    final preprocessed = await ImagePreProcessor.preprocessImage(image);

    // Step 2: Process tiles and collect raw detections
    final tileDetections =
        await tiler.processTiles(preprocessed, inferenceCallback);

    // Step 3: Post-processing (filter + NMS)
    final finalDetections = await filter.filterAndDeduplicate(tileDetections);

    return finalDetections;
  }
}
