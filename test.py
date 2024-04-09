import tensorflow as tf
import tensorflow.keras as keras


print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")