import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import tensorflow as tf
from data.data_loader import MalariaDataset

print("Loading saved model from: experiments/logs/checkpoints/mobilenet_saved")
saved_model_path = "experiments/logs/checkpoints/mobilenet_saved"
tflite_out = Path("experiments/logs/checkpoints/exports/tflite/mobilenetv3_small_int8.tflite")
tflite_out.parent.mkdir(parents=True, exist_ok=True)

print("Loading NIH dataset for calibration...")
ds = MalariaDataset(dataset_root="nih_data", image_size=(224, 224), batch_size=64, test_split=0.2, seed=42)
train_ds, _ = ds.create_datasets()

def representative_dataset():
    count = 0
    for batch in train_ds.take(100):
        images = batch[0]
        for i in range(images.shape[0]):
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count} calibration samples...")
            yield [tf.expand_dims(images[i], axis=0).numpy().astype('float32')]

print("Converting to INT8 TFLite (this may take a few minutes)...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
tflite_out.write_bytes(tflite_model)
print(f"SUCCESS: Wrote {tflite_out}")
print(f"File size: {tflite_out.stat().st_size} bytes")
