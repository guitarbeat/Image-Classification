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
        'seaborn', 'scikit-learn', 'openpyxl', 'tensorflow<2.11'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_tensorflow_gpu_support():
    """Check TensorFlow GPU support and configure GPU memory growth."""
    try:
        if tf.__version__ < '2.11':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU support is configured for TensorFlow.")
        else:
            print("Warning: TensorFlow version above 2.10 is not compatible with GPU on native Windows installations.")
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
