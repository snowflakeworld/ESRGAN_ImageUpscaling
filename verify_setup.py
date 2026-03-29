import tensorflow as tf
import sys

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

# Check for GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPUs Detected: {len(gpus)}")
    for gpu in gpus:
        print(f" - {gpu}")

    # Enable Memory Growth (Prevents TF from taking all VRAM)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth: Enabled")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs Detected. Training will run on CPU.")

# Test a simple operation
try:
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"TensorFlow Test Success: {c}")
except Exception as e:
    print(f"TensorFlow Test Failed: {e}")
