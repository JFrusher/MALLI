"""Mobile export pipeline for multi-format model serialization.

Handles export to TFLite, ONNX, and CoreML with automatic mobile asset management.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class ExportMetadata:
    """Metadata about exported model for mobile consumption."""
    
    model_format: str  # "tflite", "onnx", "coreml"
    export_timestamp: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: str
    output_dtype: str
    file_path: str
    file_size_bytes: int
    model_version: str
    framework: str = "tensorflow"
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            k: v if not isinstance(v, (tuple, list)) else list(v) 
            for k, v in asdict(self).items()
        }


class TFLiteExporter:
    """Export Keras models to INT8 quantized TFLite format."""
    
    @staticmethod
    def export(
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        output_path: Path,
        representative_batches: int = 100,
    ) -> ExportMetadata:
        """Export model to fully-quantized INT8 TFLite.
        
        Args:
            model: Trained Keras model
            train_ds: Training dataset for calibration samples
            output_path: Where to save the TFLite model
            representative_batches: Number of batches for quantization calibration
            
        Returns:
            ExportMetadata with export details
        """
        logger.info("Exporting INT8 TFLite model to %s", output_path)
        
        # Extract input/output shapes from model
        input_shape = tuple(model.input_shape[1:])
        output_shape = tuple(model.output_shape[1:])
        
        def representative_dataset_generator():
            """Yield samples for post-training integer quantization."""
            for batch in train_ds.take(representative_batches):
                images = batch[0]
                for i in range(images.shape[0]):
                    sample = tf.expand_dims(images[i], axis=0).numpy().astype(np.float32)
                    yield [sample]
        
        # Convert and quantize
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_generator
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)
        
        file_size = output_path.stat().st_size
        logger.info("Saved quantized TFLite model (%d bytes)", file_size)
        
        return ExportMetadata(
            model_format="tflite",
            export_timestamp=str(Path(output_path).parent.name),
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype="int8",
            output_dtype="int8",
            file_path=str(output_path),
            file_size_bytes=file_size,
            model_version="1.0",
        )


class ONNXExporter:
    """Export Keras models to ONNX format (for cross-platform mobile)."""
    
    @staticmethod
    def export(
        model: tf.keras.Model,
        output_path: Path,
    ) -> ExportMetadata:
        """Export model to ONNX format.
        
        Args:
            model: Trained Keras model
            output_path: Where to save the ONNX model
            
        Returns:
            ExportMetadata with export details
        """
        try:
            import tf2onnx
        except ImportError:
            logger.warning(
                "ONNX export requested but tf2onnx not installed. "
                "Install with: pip install tf2onnx onnx onnxruntime"
            )
            return None
        
        logger.info("Exporting ONNX model to %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        input_shape = tuple(model.input_shape[1:])
        output_shape = tuple(model.output_shape[1:])
        
        # Convert using tf2onnx
        import onnx
        
        spec = (tf.TensorSpec(model.input_shape, model.input.dtype),)
        output_path_str = str(output_path)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path_str)
        
        file_size = output_path.stat().st_size
        logger.info("Saved ONNX model (%d bytes)", file_size)
        
        return ExportMetadata(
            model_format="onnx",
            export_timestamp=str(Path(output_path).parent.name),
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype="float32",
            output_dtype="float32",
            file_path=str(output_path),
            file_size_bytes=file_size,
            model_version="1.0",
        )


class CoreMLExporter:
    """Export Keras models to CoreML format (for iOS)."""
    
    @staticmethod
    def export(
        model: tf.keras.Model,
        output_path: Path,
    ) -> Optional[ExportMetadata]:
        """Export model to CoreML format.
        
        Args:
            model: Trained Keras model
            output_path: Where to save the CoreML model
            
        Returns:
            ExportMetadata with export details, or None if export failed
        """
        try:
            import coremltools as ct
        except ImportError:
            logger.warning(
                "CoreML export requested but coremltools not installed. "
                "Install with: pip install coremltools (macOS only)"
            )
            return None
        
        logger.info("Exporting CoreML model to %s", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        input_shape = tuple(model.input_shape[1:])
        output_shape = tuple(model.output_shape[1:])
        
        try:
            # Convert Keras to CoreML
            mlmodel = ct.convert(
                model,
                inputs=[ct.ImageType(name="image", shape=model.input_shape)],
                classifier_config=ct.ClassifierConfig(["uninfected", "infected"]),
            )
            
            mlmodel.save(str(output_path))
            file_size = output_path.stat().st_size
            logger.info("Saved CoreML model (%d bytes)", file_size)
            
            return ExportMetadata(
                model_format="coreml",
                export_timestamp=str(Path(output_path).parent.name),
                input_shape=input_shape,
                output_shape=output_shape,
                input_dtype="image",
                output_dtype="float32",
                file_path=str(output_path),
                file_size_bytes=file_size,
                model_version="1.0",
            )
        except Exception as e:
            logger.error("CoreML export failed: %s", e)
            return None


class ExportPipeline:
    """Orchestrates multi-format model export."""
    
    def __init__(
        self,
        export_dir: Path,
        formats: list[str] | None = None,
    ):
        """Initialize export pipeline.
        
        Args:
            export_dir: Root directory for exports
            formats: List of formats to export ("tflite", "onnx", "coreml")
                    Defaults to ["tflite"]
        """
        self.export_dir = Path(export_dir)
        self.formats = formats or ["tflite"]
        self.metadata_list: list[ExportMetadata] = []
    
    def export_all(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        model_name: str = "malaria_detector",
        representative_batches: int = 100,
    ) -> dict[str, ExportMetadata]:
        """Export model to all configured formats.
        
        Args:
            model: Trained Keras model
            train_ds: Training dataset (for quantization calibration)
            model_name: Name prefix for exported models
            representative_batches: Batches for quantization calibration
            
        Returns:
            Dict mapping format -> ExportMetadata
        """
        results = {}
        
        for fmt in self.formats:
            logger.info("Starting %s export...", fmt.upper())
            
            output_path = self.export_dir / fmt / f"{model_name}.{self._get_extension(fmt)}"
            
            try:
                if fmt == "tflite":
                    metadata = TFLiteExporter.export(
                        model,
                        train_ds,
                        output_path,
                        representative_batches=representative_batches,
                    )
                elif fmt == "onnx":
                    metadata = ONNXExporter.export(model, output_path)
                elif fmt == "coreml":
                    metadata = CoreMLExporter.export(model, output_path)
                else:
                    logger.warning("Unknown export format: %s", fmt)
                    continue
                
                if metadata:
                    results[fmt] = metadata
                    self.metadata_list.append(metadata)
                    logger.info("✓ %s export successful: %s", fmt.upper(), output_path)
                
            except Exception as e:
                logger.error("✗ %s export failed: %s", fmt.upper(), e, exc_info=True)
        
        return results
    
    def save_metadata(self, output_path: Path) -> None:
        """Save export metadata as JSON.
        
        Args:
            output_path: Where to save metadata file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_dict = {
            "exports": [m.to_dict() for m in self.metadata_list],
            "count": len(self.metadata_list),
        }
        output_path.write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
        logger.info("Saved export metadata to %s", output_path)
    
    @staticmethod
    def _get_extension(fmt: str) -> str:
        """Get file extension for format."""
        extensions = {
            "tflite": "tflite",
            "onnx": "onnx",
            "coreml": "mlmodel",
        }
        return extensions.get(fmt, fmt)
