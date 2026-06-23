import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import '../database/database_helper.dart';
import '../models/sample.dart';
import '../services/tflite_service.dart';
import '../services/blood_smear_analyzer.dart';
import 'camera_capture_screen.dart';

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({Key? key}) : super(key: key);

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  final TextEditingController _idController = TextEditingController();
  bool _processing = false;

  late final TFLiteService _tfliteService;
  late final BloodSmearAnalyzer _analyzer;

  @override
  void initState() {
    super.initState();
    _tfliteService = TFLiteService();
    _analyzer = BloodSmearAnalyzer(classifier: _tfliteService);
    _tfliteService.loadModel();
  }

  @override
  void dispose() {
    _idController.dispose();
    _tfliteService.dispose();
    super.dispose();
  }

  Future<void> _openCameraAndProcess() async {
    final idTag = _idController.text.trim();
    if (idTag.isEmpty) return;

    // Open camera capture screen
    final capturedPath = await Navigator.push<String?>(
      context,
      MaterialPageRoute(builder: (_) => const CameraCaptureScreen()),
    );
    if (capturedPath == null) return;

    setState(() => _processing = true);

    try {
      // Copy the captured file into the app documents directory for persistence
      final docs = await getApplicationDocumentsDirectory();
      final filename = p.basename(capturedPath);
      final newPath = p.join(docs.path, 'images', '${DateTime.now().millisecondsSinceEpoch}_$filename');
      final imagesDir = Directory(p.join(docs.path, 'images'));
      if (!await imagesDir.exists()) await imagesDir.create(recursive: true);
      final newFile = await File(capturedPath).copy(newPath);

      // Insert DB entry with pending status
      final sample = Sample(idTag: idTag, imagePath: newFile.path, status: SampleStatus.processing);
      final id = await DatabaseHelper.instance.insertSample(sample);

      // Run on-device blood smear analysis
      final result = await _analyzer.analyze(newFile.path);

      // Mark completed and delete image file to save space
      await DatabaseHelper.instance.markCompletedAndDeleteImage(
        id,
        result.parasitemiaPercent,
        newFile.path,
        totalCells: result.totalCells,
        infectedCells: result.infectedCells,
      );
    } catch (e) {
      // Handle/log error; keep simple for scaffold
    } finally {
      setState(() => _processing = false);
      if (mounted) Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Capture Sample')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _idController,
              decoration: const InputDecoration(labelText: 'ID Tag (manual entry)'),
            ),
            const SizedBox(height: 16),
            _processing
                ? const CircularProgressIndicator()
                : ElevatedButton.icon(
                    onPressed: _openCameraAndProcess,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Open Camera and Capture'),
                  ),
          ],
        ),
      ),
    );
  }
}
