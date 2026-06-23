import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Loads and runs the exported MobileNetV3-small INT8 TFLite model.
///
/// Handles both float32 and INT8 quantized models transparently via
/// per-tensor quantization parameters read from the model at load time.
class TFLiteService {
  static const int _inputSize = 224;
  static const String _defaultAssetPath = 'assets/models/malaria_detector.tflite';

  Interpreter? _interpreter;
  bool _isInt8 = false;

  // INT8 input quantization parameters (scale, zeroPoint)
  double _inputScale = 1.0;
  int _inputZeroPoint = 0;

  // INT8 output quantization parameters
  double _outputScale = 1.0;
  int _outputZeroPoint = 0;

  bool get isLoaded => _interpreter != null;

  /// Loads the TFLite model from Flutter assets.
  ///
  /// Call once before any inference. Reads quantization parameters so
  /// [classifyCell] can quantize/dequantize transparently for INT8 models.
  Future<void> loadModel([String assetPath = _defaultAssetPath]) async {
    _interpreter?.close();

    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset(assetPath, options: options);

    final inputTensor = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);

    // TensorType.int8 == 9 in tflite_flutter enum
    _isInt8 = inputTensor.type == TensorType.int8;

    if (_isInt8) {
      _inputScale = inputTensor.params.scale;
      _inputZeroPoint = inputTensor.params.zeroPoint;
      _outputScale = outputTensor.params.scale;
      _outputZeroPoint = outputTensor.params.zeroPoint;
    }
  }

  /// Loads the TFLite model from a file path (useful for side-loading).
  Future<void> loadModelFromFile(String filePath) async {
    _interpreter?.close();

    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromFile(File(filePath), options: options);

    final inputTensor = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);

    _isInt8 = inputTensor.type == TensorType.int8;

    if (_isInt8) {
      _inputScale = inputTensor.params.scale;
      _inputZeroPoint = inputTensor.params.zeroPoint;
      _outputScale = outputTensor.params.scale;
      _outputZeroPoint = outputTensor.params.zeroPoint;
    }
  }

  /// Classifies a single cell crop image.
  ///
  /// Resizes [cellCrop] to 224×224, normalises pixels to [0,1], applies
  /// INT8 quantization if the loaded model requires it, runs inference, then
  /// dequantizes the output and returns a probability in [0.0, 1.0] where
  /// 1.0 = Parasitized.
  ///
  /// Throws [StateError] if the model has not been loaded yet.
  Future<double> classifyCell(img.Image cellCrop) async {
    if (_interpreter == null) {
      throw StateError('TFLiteService: model not loaded — call loadModel() first');
    }

    // Resize to model input size
    final resized = img.copyResize(cellCrop, width: _inputSize, height: _inputSize);

    if (_isInt8) {
      return _runInt8Inference(resized);
    } else {
      return _runFloat32Inference(resized);
    }
  }

  double _runFloat32Inference(img.Image image) {
    // Build float32 input tensor [1, 224, 224, 3]
    final input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) {
            final pixel = image.getPixel(x, y);
            return [
              pixel.r / 255.0,
              pixel.g / 255.0,
              pixel.b / 255.0,
            ];
          },
        ),
      ),
    );

    final output = List.generate(1, (_) => [0.0]);
    _interpreter!.run(input, output);
    return (output[0][0]).clamp(0.0, 1.0);
  }

  double _runInt8Inference(img.Image image) {
    // Pack pixels into a flat Int8List: NHWC layout [1, 224, 224, 3]
    final inputBytes = Int8List(_inputSize * _inputSize * 3);
    int idx = 0;
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = image.getPixel(x, y);
        inputBytes[idx++] = _quantize(pixel.r / 255.0);
        inputBytes[idx++] = _quantize(pixel.g / 255.0);
        inputBytes[idx++] = _quantize(pixel.b / 255.0);
      }
    }

    final outputBytes = Int8List(1);

    // Reshape buffers for tflite_flutter's run() method
    final reshapedInput = inputBytes.reshape([1, _inputSize, _inputSize, 3]);
    final reshapedOutput = outputBytes.reshape([1, 1]);

    _interpreter!.run(reshapedInput, reshapedOutput);

    // Dequantize single int8 output to float probability
    final rawInt8 = outputBytes[0];
    final probability = (rawInt8 - _outputZeroPoint) * _outputScale;
    return probability.clamp(0.0, 1.0);
  }

  /// Quantizes a float [0,1] value to int8 using model-derived scale/zeroPoint.
  int _quantize(double floatVal) {
    final q = (floatVal / _inputScale + _inputZeroPoint).round();
    return q.clamp(-128, 127);
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
