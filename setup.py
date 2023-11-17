### System Setup and Optimization for TensorFlow

import os
import sys
import subprocess
import platform
import tensorflow as tf
import pkg_resources

def is_package_installed(package_name):
    """Check if a package is already installed."""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False
    
def install_packages():
    """Install required packages if not already installed."""
    packages = [
        'opencv-python', 'numpy', 'pandas', 'matplotlib', 'protobuf', 
        'seaborn', 'scikit-learn', 'openpyxl', 'tensorflow<2.11',
        'pydot', 'pydotplus', 'graphviz', 'scipy', 'keras', 'keras-core',
        'jax', 'jaxlib', 'jax[cpu]','tensorboard','jinja2','protobuf',
    ]
    for package in packages:
        if not is_package_installed(package):
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
            if any(gpu.device_type == 'GPU' for gpu in gpus):
                #  and gpu.compute_capability >= 7.0
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision policy set to 'mixed_float16'.")
            else:
                print("Warning: Mixed precision may run slowly as GPU compute capability is less than 7.0.")
        else:
            print("No GPU detected. Mixed precision will not be enabled.")
    except Exception as e:
        print(f"Error configuring TensorFlow GPU support: {e}")

def print_system_info():
    """Print system and TensorFlow information."""
    system_info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "TensorFlow Version": tf.__version__,
        "Num GPUs Available": len(tf.config.list_physical_devices('GPU')),
    }
    formatted_info = "\n".join(f"{key}: {value}" for key, value in system_info.items())
    print(formatted_info)


def setup():
    """Main setup function to be called at the start of the notebook."""
    install_packages()
    check_tensorflow_gpu_support()
    print_system_info()

if __name__ == "__main__":
    setup()
