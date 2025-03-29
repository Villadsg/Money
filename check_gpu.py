#!/usr/bin/env python3
"""
Check if TensorFlow can detect available GPUs
"""

import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)
print("\nGPU Information:")
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# List physical devices
print("\nPhysical Devices:")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(f"  {device.device_type}: {device.name}")

# Check if CUDA is available
print("\nCUDA Available:", tf.test.is_built_with_cuda())

# Try to run a simple operation on GPU
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("\nMatrix multiplication result:")
        print(c)
        print("Operation executed successfully on GPU")
except RuntimeError as e:
    print("\nFailed to execute operation on GPU:", str(e))

# Print environment variables related to CUDA
print("\nEnvironment Variables:")
for var in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']:
    print(f"  {var}: {os.environ.get(var, 'Not set')}")
