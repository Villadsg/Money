#!/usr/bin/env python3
"""
Detailed CUDA configuration check
"""

import os
import subprocess
import sys
import platform

print("=" * 50)
print("SYSTEM INFORMATION")
print("=" * 50)
print(f"Python version: {platform.python_version()}")
print(f"OS: {platform.system()} {platform.release()}")

print("\n" + "=" * 50)
print("CUDA CONFIGURATION")
print("=" * 50)

# Check NVIDIA driver
try:
    nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode('utf-8')
    print("\nNVIDIA Driver Information:")
    print(nvidia_smi.split('\n')[0:3])
except Exception as e:
    print(f"Error checking NVIDIA driver: {e}")

# Check CUDA version
try:
    nvcc_output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode('utf-8')
    print("\nCUDA Compiler (nvcc) Version:")
    print(nvcc_output)
except Exception as e:
    print(f"Error checking CUDA version: {e}")

# Check TensorRT installation
print("\nChecking for TensorRT libraries:")
tensorrt_libs = [
    "libnvinfer.so.7",
    "libnvinfer_plugin.so.7"
]
for lib in tensorrt_libs:
    # Use ldconfig to check if the library is in the system's library path
    result = subprocess.run(['ldconfig', '-p'], stdout=subprocess.PIPE, text=True)
    if lib in result.stdout:
        print(f"✓ {lib} is installed")
    else:
        print(f"✗ {lib} is NOT installed")

# Check environment variables
print("\nEnvironment Variables:")
cuda_env_vars = [
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH"
]
for var in cuda_env_vars:
    value = os.environ.get(var, "Not set")
    print(f"{var}: {value}")

# Look for CUDA installation directories
print("\nSearching for CUDA installation directories:")
common_cuda_paths = [
    "/usr/local/cuda",
    "/usr/lib/cuda",
    "/opt/cuda"
]
for path in common_cuda_paths:
    if os.path.exists(path):
        print(f"Found CUDA installation: {path}")
        # List key subdirectories
        for subdir in ['lib64', 'include', 'bin']:
            full_path = os.path.join(path, subdir)
            if os.path.exists(full_path):
                print(f"  - {full_path} exists")
            else:
                print(f"  - {full_path} does NOT exist")

print("\n" + "=" * 50)
print("LIBRARY LOAD TEST")
print("=" * 50)
try:
    import ctypes
    cuda_runtime = ctypes.CDLL("libcudart.so")
    print("Successfully loaded CUDA Runtime library")
except Exception as e:
    print(f"Failed to load CUDA Runtime library: {e}")

print("\n" + "=" * 50)
print("RECOMMENDATIONS")
print("=" * 50)
