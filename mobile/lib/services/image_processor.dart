import 'dart:async';
import 'dart:io';
import 'package:image/image.dart' as img;
import 'detection_pipeline.dart';

/// ============================================================================
/// IMAGE LOADING & PREPROCESSING
/// ============================================================================

/// Loads image from file path into memory-efficient format.
/// 
/// Handles common image formats (PNG, JPG, etc.) and returns null on error.
Future<img.Image?> loadImageFromPath(String path) async {
  try {
    final file = File(path);
    if (!await file.exists()) {
      print('Image file not found: $path');
      return null;
    }

    final bytes = await file.readAsBytes();
    final image = img.decodeImage(bytes);
    return image;
  } catch (e) {
    print('Error loading image: $e');
    return null;
  }
}

/// ============================================================================
/// HIGH-LEVEL ANALYSIS API
/// ============================================================================

/// Complete blood smear analysis pipeline using the detection classes.
/// 
/// This is the main entry point for mobile analysis. It orchestrates:
/// 1. Load and preprocess image
/// 2. Apply SAHI-style tiling
/// 3. Run inference on tiles (via callback)
/// 4. Post-process detections (geometric + NMS)
/// 5. Calculate parasitemia percentage
/// 
/// [imagePath]: Path to blood smear image file
/// [customInferenceCallback]: Optional custom inference function.
///                            If not provided, uses placeholder.
/// [tileSize]: Model input size (default 640 for YOLOv8)
/// [overlapRatio]: Overlap between tiles (default 0.2 = 20%)
/// 
/// Returns: [BloodSmearAnalysisResult] with detected cells and metrics
Future<BloodSmearAnalysisResult> analyzeBloodSmear(
  String imagePath, {
  Future<List<BoundingBox>> Function(img.Image)? customInferenceCallback,
  int tileSize = 640,
  double overlapRatio = 0.2,
}) async {
  // Step 1: Load image
  final image = await loadImageFromPath(imagePath);
  if (image == null) {
    throw Exception('Failed to load image from $imagePath');
  }

  // Step 2: Initialize pipeline components
  final tiler = TilingInferenceEngine(
    tileSize: tileSize,
    overlapRatio: overlapRatio,
  );

  final filter = PostInferenceFilter(
    minArea: 100, // Minimum cell area (pixels)
    maxAreaRatio: 0.15, // Max 15% of tile area
    nmsThreshold: 0.5, // IoU threshold for NMS
    minAspectRatio: 0.3,
    maxAspectRatio: 3.0,
    tileSize: tileSize,
  );

  final pipeline = BloodSmearDetectionPipeline(
    tiler: tiler,
    filter: filter,
  );

  // Step 3: Use provided inference callback or default placeholder
  final inferenceCallback = customInferenceCallback ??
      (tileImage) async {
        // Placeholder: return empty detections
        // In production, this would call TFLite YOLO model
        await Future.delayed(const Duration(milliseconds: 50));
        return [];
      };

  // Step 4: Run detection pipeline
  final detections = await pipeline.detectCells(image, inferenceCallback);

  // Step 5: Calculate statistics
  final parasitemiaPercent = detections.isNotEmpty
      ? (detections.where((d) => d.confidence > 0.5).length /
              detections.length *
              100)
          .toDouble()
      : 0.0;

  return BloodSmearAnalysisResult(
    detectedCells: detections,
    parasitemiaPercent: parasitemiaPercent,
    totalCellsAnalyzed: detections.length,
    imagePath: imagePath,
  );
}

/// Placeholder analysis function for legacy compatibility.
/// 
/// Returns dummy parasitemia percentage.
/// This is maintained for backward compatibility with existing UI code.
/// 
/// **Deprecated**: Use [analyzeBloodSmear] instead.
Future<double> processImage(String path) async {
  await Future.delayed(const Duration(seconds: 2));
  // Return dummy percentage
  return 5.5;
}

/// ============================================================================
/// RESULT OBJECT
/// ============================================================================

/// Result object from blood smear analysis.
/// 
/// Contains all detection results and calculated metrics.
class BloodSmearAnalysisResult {
  /// List of detected parasites/cells with coordinates and confidence
  final List<BoundingBox> detectedCells;

  /// Calculated parasitemia percentage (0-100)
  final double parasitemiaPercent;

  /// Total number of cells analyzed
  final int totalCellsAnalyzed;

  /// Path to analyzed image
  final String imagePath;

  /// Analysis timestamp
  final DateTime timestamp;

  BloodSmearAnalysisResult({
    required this.detectedCells,
    required this.parasitemiaPercent,
    required this.totalCellsAnalyzed,
    required this.imagePath,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// Get high-confidence detections (confidence > 0.5)
  List<BoundingBox> get highConfidenceDetections =>
      detectedCells.where((d) => d.confidence > 0.5).toList();

  /// Get medium-confidence detections (0.3 < confidence <= 0.5)
  List<BoundingBox> get mediumConfidenceDetections => detectedCells
      .where((d) => d.confidence > 0.3 && d.confidence <= 0.5)
      .toList();

  /// Convert result to JSON for storage/transmission
  Map<String, dynamic> toJson() {
    return {
      'parasitemia_percent': parasitemiaPercent,
      'total_cells': totalCellsAnalyzed,
      'high_confidence_cells': highConfidenceDetections.length,
      'timestamp': timestamp.toIso8601String(),
      'detections': detectedCells
          .map((d) => {
                'x': d.x,
                'y': d.y,
                'width': d.width,
                'height': d.height,
                'confidence': d.confidence,
                'classId': d.classId,
              })
          .toList(),
    };
  }

  @override
  String toString() =>
      'BloodSmearAnalysisResult(cells: $totalCellsAnalyzed, parasitemia: ${parasitemiaPercent.toStringAsFixed(1)}%, timestamp: $timestamp)';
}
