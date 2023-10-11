# utilities.py

# ------------------------------
# System and TensorFlow Info Check
# ------------------------------
import platform
import tensorflow as tf

def get_system_info():
    """Get system and TensorFlow information."""
    system_info = {"Platform": platform.platform(), "Python Version": platform.python_version()}
    
    try:
        system_info.update({
            "TensorFlow Version": tf.__version__,
            "Num GPUs Available": len(tf.config.list_physical_devices('GPU'))
        })
        system_info['Instructions'] = (
            "You're all set to run your model on a GPU." 
            if system_info['Num GPUs Available'] 
            else (
                "No GPUs found. To use a GPU, follow these steps:\n"
                "  1. Install NVIDIA drivers for your GPU.\n"
                "  2. Install a compatible CUDA toolkit.\n"
                "  3. Install the cuDNN library.\n"
                "  4. Make sure to install the GPU version of TensorFlow."
            )
        )
    except ModuleNotFoundError:
        system_info['Instructions'] = (
            "TensorFlow is not installed. "
            "Install it using pip by running: !pip install tensorflow"
        )
    
    return system_info

def configure_gpu_memory_growth():
    """Set GPU memory consumption growth to avoid OOM errors."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Call functions to get system info and configure GPU memory.
system_info = get_system_info()
configure_gpu_memory_growth()

# Print system information.
formatted_info = "\n".join(f"{key}: {value}" for key, value in system_info.items())
print(formatted_info)

