# model_management.py

import re

def get_last_saved_model(model_dir):
    # Get all the saved models
    saved_models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not saved_models:
        return None, 0  # No saved model found, start from scratch

    # Identify the last epoch from filename (assuming you saved it as part of the filename)
    last_model, last_epoch = max(saved_models, key=os.path.getctime), 0  # Just an initial setup

    # Extract epoch number from the filename; 
    # depends on your naming scheme, adjust the regular expression accordingly
    for model_file in saved_models:
        epoch_search = re.search(r'epoch_(\d+)', model_file)
        if epoch_search:
            epoch = int(epoch_search[1])
            if epoch > last_epoch:
                last_epoch = epoch
                last_model = model_file

    return os.path.join(model_dir, last_model), last_epoch


### Loading the Best Model from Directories

def get_best_model_filename(directory):
    """Identify the best model filename based on the minimum validation loss from the directory."""
    model_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not model_files:
        print(f"No model files found in {directory}")
        return None
    return min(model_files, key=lambda x: float(x.split('val_loss_')[1].split('_')[0]))

def load_best_model(directory):
    """Loads the best model from the specified directory."""
    best_model_file = get_best_model_filename(directory)
    if not best_model_file:
        return None
    best_model_path = os.path.join(directory, best_model_file)
    # return load_model(best_model_path)
    return load_model(best_model_path, compile=False)

def get_label_directories(experiment_directory):
    """Determine label directories or just model directories in the experiment directory."""
    first_level_dirs = [os.path.join(experiment_directory, d) for d in os.listdir(experiment_directory) 
                        if os.path.isdir(os.path.join(experiment_directory, d))]
    if any('mobilenetv2' in dir_name for dir_name in first_level_dirs):
        return [experiment_directory]
    return first_level_dirs

def load_all_best_models(experiment_directory):
    """Load the best model for each model type within the experiment directory."""
    best_models = {}
    label_dirs = get_label_directories(experiment_directory)
    for label_dir in label_dirs:
        for model_name in ['mobilenetv2', 'inceptionv3', 'resnet50', 'small_xception', 'basic_cnn']:
            model_dir = os.path.join(label_dir, model_name)
            best_model = load_best_model(model_dir)
            if best_model:
                key_name = f"{os.path.basename(label_dir)}_{model_name}"
                best_models[key_name] = best_model
    return best_models

# Example Usage
experiment_directory = "SIM_Unbalanced"
all_best_models = load_all_best_models(experiment_directory)
print(all_best_models.keys())  # This will display the keys of the loaded models.

