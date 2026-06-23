import 'dart:io';
import 'package:image/image.dart' as img;
import 'detection_pipeline.dart';
import 'tflite_service.dart';

/// Result of a full blood smear analysis run.
class BloodSmearResult {
  final int totalCells;
  final int infectedCells;
  final double parasitemiaPercent;
  final List<BoundingBox> detections;
  final DateTime timestamp;

  BloodSmearResult({
    required this.totalCells,
    required this.infectedCells,
    required this.parasitemiaPercent,
    required this.detections,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  Map<String, dynamic> toJson() => {
        'totalCells': totalCells,
        'infectedCells': infectedCells,
        'parasitemiaPercent': parasitemiaPercent,
        'timestamp': timestamp.toIso8601String(),
        'detections': detections
            .map((d) => {
                  'x': d.x,
                  'y': d.y,
                  'width': d.width,
                  'height': d.height,
                  'confidence': d.confidence,
                })
            .toList(),
      };

  @override
  String toString() =>
      'BloodSmearResult(total: $totalCells, infected: $infectedCells, '
      'parasitemia: ${parasitemiaPercent.toStringAsFixed(2)}%)';
}

/// Dart implementation of the on-device blood smear analysis pipeline.
///
/// Mirrors the Python CellCounter workflow:
///   load image → green-channel enhance → tile → classify via TFLiteService
///   → NMS → compute parasitemia %
///
/// Usage:
/// ```dart
/// final service = TFLiteService();
/// await service.loadModel();
/// final analyzer = BloodSmearAnalyzer(classifier: service);
/// final result = await analyzer.analyze('/path/to/smear.jpg');
/// ```
class BloodSmearAnalyzer {
  final TFLiteService _classifier;

  /// Probability threshold above which a cell is counted as infected.
  /// Matches the Python calibration default of 0.3.
  final double threshold;

  /// Side length (px) of each cell ROI crop fed to the classifier.
  final int roiSize;

  /// Overlap ratio between adjacent ROI tiles (0.0–1.0).
  final double overlapRatio;

  /// IoU threshold for NMS deduplication of overlapping detections.
  final double nmsIoUThreshold;

  BloodSmearAnalyzer({
    required TFLiteService classifier,
    this.threshold = 0.3,
    this.roiSize = 128,
    this.overlapRatio = 0.2,
    this.nmsIoUThreshold = 0.5,
  }) : _classifier = classifier;

  /// Runs the full blood smear analysis pipeline on [imagePath].
  ///
  /// Throws [FileSystemException] if the image cannot be read.
  /// Throws [StateError] if [TFLiteService] has not been loaded.
  Future<BloodSmearResult> analyze(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    final raw = img.decodeImage(bytes);
    if (raw == null) {
      throw Exception('BloodSmearAnalyzer: failed to decode image at $imagePath');
    }

    // Step 1: green-channel enhancement (matches Python green-channel CLAHE step)
    final enhanced = await ImagePreProcessor.preprocessImage(raw);

    // Step 2: generate overlapping ROI tiles across the smear
    final tiler = TilingInferenceEngine(tileSize: roiSize, overlapRatio: overlapRatio);
    final tiles = tiler.generateTiles(enhanced.width, enhanced.height);

    // Step 3: classify each tile, collect detections above threshold
    final rawDetections = <BoundingBox>[];

    for (final tile in tiles) {
      final tileX = tile['x']!;
      final tileY = tile['y']!;

      final cellCrop = await TilingInferenceEngine.extractTile(
        enhanced,
        tileX,
        tileY,
        tile['width']!,
        tile['height']!,
        roiSize,
      );

      final probability = await _classifier.classifyCell(cellCrop);

      // Every tile is a candidate cell; confidence = the model's probability
      rawDetections.add(BoundingBox(
        x: tileX.toDouble(),
        y: tileY.toDouble(),
        width: roiSize.toDouble(),
        height: roiSize.toDouble(),
        confidence: probability,
        classId: probability >= threshold ? 1 : 0,
      ));
    }

    // Step 4: NMS to remove duplicate detections of the same cell
    final filter = PostInferenceFilter(
      minArea: 0,                    // all tiles are valid size
      maxAreaRatio: 1.0,             // no area cap on ROI tiles
      nmsThreshold: nmsIoUThreshold,
      minAspectRatio: 0.5,
      maxAspectRatio: 2.0,
      tileSize: roiSize,
    );
    final deduplicated = filter.applyNMS(rawDetections);

    // Step 5: tally results
    final infectedCells = deduplicated.where((d) => d.classId == 1).length;
    final totalCells = deduplicated.length;
    final parasitemiaPercent =
        totalCells > 0 ? (infectedCells / totalCells) * 100.0 : 0.0;

    return BloodSmearResult(
      totalCells: totalCells,
      infectedCells: infectedCells,
      parasitemiaPercent: parasitemiaPercent,
      detections: deduplicated,
    );
  }
}
