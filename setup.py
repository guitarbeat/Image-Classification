# setup.py

import os
import sys
import subprocess
import platform
import tensorflow as tf



def install_packages():
    """
    Install required packages using pip.
    
    These packages are commonly used for data handling, machine learning, and visualization.
    opencv-python: For image processing tasks.
    numpy: Fundamental package for scientific computing with Python.
    pandas: Data analysis and manipulation tool.
    matplotlib: Plotting library for Python.
    seaborn: Statistical data visualization based on matplotlib.
    scikit-learn: Machine learning library.
    openpyxl: A Python library to read/write Excel 2010 xlsx/xlsm files.
    """
    packages = [
        'opencv-python', 'numpy', 'pandas', 'matplotlib', 'protobuf', 
        'seaborn', 'scikit-learn', 'openpyxl', 'tensorflow<2.11',
        'pydot',
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_tensorflow_gpu_support():
    """Check TensorFlow GPU support and configure GPU memory growth and mixed precision."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU support is configured for TensorFlow.")
            
            # Check for GPU compute capability for mixed precision
            if any(gpu.device_type == 'GPU' and gpu.compute_capability >= 7.0 for gpu in gpus):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision policy set to 'mixed_float16'.")
            else:
                print("Warning: Mixed precision may run slowly as GPU compute capability is less than 7.0.")
        else:
            print("No GPU detected. Mixed precision will not be enabled.")
    except Exception as e:
        print(f"Error configuring TensorFlow GPU support: {e}")

def get_system_info():
    """Get system and TensorFlow information."""
    return {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "TensorFlow Version": tf.__version__,
        "Num GPUs Available": len(tf.config.list_physical_devices('GPU')),
    }

def print_system_info():
    """Print system information."""
    system_info = get_system_info()
    formatted_info = "\n".join(f"{key}: {value}" for key, value in system_info.items())
    print(formatted_info)



def setup():
    """Main setup function to be called at the start of the notebook."""
    install_packages()
    check_tensorflow_gpu_support()
    print_system_info()

if __name__ == "__main__":
    setup()
