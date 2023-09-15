# ------------------------------
# Package Installation (Optional)
# ------------------------------
# Uncomment the following lines to install required packages if running on a new machine.
# To suppress the output, we use '> /dev/null 2>&1'.
# %pip install numpy pandas matplotlib protobuf seaborn scikit-learn tensorflow > /dev/null 2>&1

# ------------------------------
# Import Libraries
# ------------------------------

# Standard Libraries
import os, sys, random, math, glob, logging
from datetime import datetime
from collections import defaultdict

# Third-Party Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from IPython.display import clear_output
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.applications import InceptionV3, ResNet50
from keras.models import load_model
from tensorflow.data import Dataset

import pickle

# Type Annotations
from typing import List, Dict, Tuple, Union, Any, Optional
# Configuration dictionary
config = {
    'Experiment': {
        'NAME': "Multi-Label_Thresholds-30-60-1-2",  # Experiment name
        'RANDOM_SEED': 42,  # Seed for reproducibility
        'PROBLEM_TYPE': 'Multi-Class',  # Problem type: Binary, Multi-Class, Multi-Label
    },
    'Model': {
        'IMG_SIZE': 224,  # Image input size
        'BATCH_SIZE': 32,  # Batch size for training
        'EPOCHS': 100,  # Number of epochs
        'LEARNING_RATE': 1e-3,  # Learning rate
        'EARLY_STOPPING_PATIENCE': 5,  # Early stopping patience parameter
        'REDUCE_LR_PATIENCE': 3,  # Learning rate reduction patience parameter
        'MIN_LR': 1e-6,  # Minimum learning rate
        'LOSS': "binary_crossentropy",  # Loss function: "categorical_crossentropy" for multi-class
        'TRAIN_SIZE': 0.8,  # Fraction of data to use for training
        'VAL_SIZE': 0.5,  # Fraction of data to use for validation
    },
    'Labels': {
        'MAPPINGS': {  # Class label mappings
            'Focus_Label': {'SharpFocus': 0, 'SlightlyBlurred': 1, 'HighlyBlurred': 2},
            'StigX_Label': {'OptimalStig_X': 0, 'ModerateStig_X': 1, 'SevereStig_X': 2},
            'StigY_Label': {'OptimalStig_Y': 0, 'ModerateStig_Y': 1, 'SevereStig_Y': 2},
        }
    },
    'Augmentation': {  # Data augmentation parameters
        'rotation_factor': 0.002,
        'height_factor': (-0.18, 0.18),
        'width_factor': (-0.18, 0.18),
        'contrast_factor': 0.5,
    }
}

# Set random seed for reproducibility
np.random.seed(config['Experiment']['RANDOM_SEED'])
tf.random.set_seed(config['Experiment']['RANDOM_SEED'])
# Read the data
def read_csv(config: Dict):
    # Functionality to read the data
    data_file_path = os.path.join(config['Paths']['NEW_BASE_PATH'], config['Paths']['DATA_FILE'])
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Error: File does not exist - {data_file_path}")
    try:
        data = pd.read_csv(data_file_path, usecols=config['CSV']['COLUMNS_TO_READ'])
        print("---> Data read successfully.")
        sample_frac = config.get('SAMPLE_FRAC', 1.0)
        if 0 < sample_frac < 1.0:
            data = data.sample(frac=sample_frac).reset_index(drop=True)
            print(f"---> Data sampled: Using {sample_frac * 100}% of the available data.")
    except Exception as e:
        raise ValueError(f"Error: Could not read data - {e}") from e
    return data

def clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    invalid_rows = []
    
    for index, row in df.iterrows():
        image_path = row['ImageFile']
        
        # Check if image_path is not string
        if not isinstance(image_path, str):
            print(f"Removing row: {row} (Reason: Invalid ImageFile value - not a string)")
            invalid_rows.append(index)
            continue
        
        # Check if the image path exists
        if not os.path.exists(image_path):
            print(f"Removing row: {row} (Reason: File does not exist)")
            invalid_rows.append(index)
            continue
        
        # Check if image can be read
        img = cv2.imread(image_path)
        if img is None:
            print(f"Removing row: {row} (Reason: Image can't be read)")
            invalid_rows.append(index)
    
    # Drop invalid rows
    df.drop(index=invalid_rows, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def update_image_paths(df):
    old_base_path = config['Paths']['OLD_BASE_PATH']
    new_base_path = config['Paths']['NEW_BASE_PATH']
    df['ImageFile'] = df['ImageFile'].str.replace(old_base_path, new_base_path, regex=False)
    print("---> Image paths updated.")
    return df

def generate_labels(df: pd.DataFrame):
    """Generate labels based on the configuration."""
    print("---> Generating labels for Focus, StigX, and StigY...")
    # Extract configurations
    labels_config = config.get('Labels', {}).get('MAPPINGS', {})
    thresholds_config = config.get('Thresholds', {})
    # Offset columns mapping
    offset_column_mapping = {
        'Focus_Label': 'Focus_Offset (V)',
        'StigX_Label': 'Stig_Offset_X (V)',
        'StigY_Label': 'Stig_Offset_Y (V)'
    }
    df_copy = df.copy()
    
    label_encoders = {}  # To store label encoders
    mlb_classes = None  # To store classes of MultiLabelBinarizer
    
    for label_key, choices_dict in labels_config.items():
        offset_column = offset_column_mapping.get(label_key)
        if not offset_column:
            print(f"Warning: No offset column mapping found for '{label_key}'. Skipping label generation.")
            continue
        if offset_column not in df.columns:
            print(f"Warning: Column '{offset_column}' not found in DataFrame. Skipping label generation for '{label_key}'.")
            continue
        
        low_threshold = thresholds_config.get(f"{label_key.split('_')[0].upper()}_LOW", 0)
        high_threshold = thresholds_config.get(f"{label_key.split('_')[0].upper()}_HIGH", 0)
        conditions = [
            (df_copy[offset_column].abs() <= low_threshold),
            (df_copy[offset_column].abs() > low_threshold) & (df_copy[offset_column].abs() <= high_threshold),
            (df_copy[offset_column].abs() > high_threshold)
        ]
        choices = list(choices_dict.keys())
        df_copy[label_key] = np.select(conditions, choices, default='Unknown')
        le = LabelEncoder()
        df_copy[label_key] = le.fit_transform(df_copy[label_key])
        label_encoders[label_key] = le
        print("---> Labels generated for", label_key)
        
    # For multi-label problems
    if config.get('Experiment', {}).get('PROBLEM_TYPE') == 'Multi-Label':
        label_keys = list(labels_config.keys())
        df_copy['Multi_Labels'] = df_copy.apply(lambda row: [row[key] for key in label_keys], axis=1)
        print("---> Multi-labels generated.")
        mlb = MultiLabelBinarizer()
        df_copy['Multi_Labels_Binarized'] = list(mlb.fit_transform(df_copy['Multi_Labels']))
        mlb_classes = mlb.classes_  # Store the classes attribute for later use
        
    return df_copy, label_encoders, mlb_classes

def shuffle_and_reset_index(data):
    print("---> Shuffling and resetting index...")
    shuffled_df = data.sample(frac=1, random_state=config['Experiment']['RANDOM_SEED']).reset_index(drop=True)
    print("---> Data shuffled and index reset.")
    return shuffled_df

def prepare_datasets(df: pd.DataFrame):
    """Prepare training, validation, and test datasets."""
    # Check if DataFrame is empty
    if df is None or df.empty:
        print("Warning: DataFrame is empty. Cannot proceed with data preparation.")
        return {'train': None, 'valid': None, 'test': None}
    # Split Data
    try:
        train_df, temp_df = train_test_split(df, test_size=1 - config['Model']['TRAIN_SIZE'], random_state=config['Experiment']['RANDOM_SEED'])
        val_df, test_df = train_test_split(temp_df, test_size=1 - config['Model']['VAL_SIZE'], random_state=config['Experiment']['RANDOM_SEED'])
    except ValueError:
        print("Not enough data to split into training, validation, and test sets.")
        return {'train': None, 'valid': None, 'test': None}
    print("---> Data split into training, validation, and test sets.")
    return {'train': train_df, 'valid': val_df, 'test': test_df}
# Compute class weights

def compute_and_store_class_weights(datasets: Dict[str, pd.DataFrame], 
                                    label_encoders: Dict[str, LabelEncoder], 
                                    mlb_classes: np.ndarray = None) -> pd.DataFrame:
    problem_type = config.get('Experiment', {}).get('PROBLEM_TYPE', 'Binary')
    
    all_records = []  # To store records before converting them to a DataFrame
    
    if problem_type == 'Multi-Label':
        mlb = MultiLabelBinarizer(classes=mlb_classes)  # Initialize with known classes if available
        for split, df in datasets.items():
            if df is None:
                continue
            
            label_column = np.array(df['Multi_Labels'].tolist())
            binarized_labels = mlb.transform(label_column)  # Use transform instead of fit_transform to ensure consistent classes
            
            for label_idx, label_name in enumerate(mlb.classes_):
                label_data = binarized_labels[:, label_idx]
                unique_labels = np.unique(label_data)
                
                class_weights = compute_class_weight('balanced', classes=unique_labels, y=label_data)
                class_weights_dict = dict(zip(unique_labels, class_weights))
                
                for cls, weight in class_weights_dict.items():
                    cnt = Counter(label_data)[cls]
                    all_records.append({'split': split, 'label': label_name, 'class': cls, 'Count': cnt, 'Weight': weight})
    
    else:  # Multi-Class or Binary
        for split, df in datasets.items():
            if df is None:
                continue
            for label in config['Labels']['MAPPINGS']:
                unique_labels = df[label].unique()
                class_weights = compute_class_weight('balanced', classes=unique_labels, y=df[label])
                
                class_weights_dict = dict(zip(unique_labels, class_weights))
                
                for cls, weight in class_weights_dict.items():
                    cnt = Counter(df[label])[cls]
                    
                    # Reverse map to original class using label_encoders
                    original_class = label_encoders[label].inverse_transform([cls])[0]
                    
                    all_records.append({'split': split, 'label': label, 'class': original_class, 'Count': cnt, 'Weight': weight})
                    
    df_class_weights = pd.DataFrame.from_records(all_records)
    df_class_weights.set_index(['split', 'label', 'class'], inplace=True)
    
    return df_class_weights
def drop_na_rows(df: pd.DataFrame, columns: list, label: str, split: str):
    """Drop rows with NA values in specified columns."""
    for column in columns:
        if df[column].isna().any():
            print(f"[WARNING] Removing rows with nan in '{column}' column for label {label} and split {split}")
            df = df.dropna(subset=[column])

def create_tf_dataset(df: pd.DataFrame, columns: Tuple[str, str], is_training: bool) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a DataFrame."""
    drop_na_rows(df, [columns[0]], columns[1], 'any')
    ds = tf.data.Dataset.from_tensor_slices((df[columns[0]].values, df[columns[1]].values))
    return preprocess_single_dataset(ds, is_training)

def preprocess_single_dataset(ds, is_training: bool = False) -> tf.data.Dataset:
    """Apply preprocessing to a single dataset."""
    return ds.map(lambda file_path, label: preprocess_wrapper(file_path, label, is_training))

def determine_label_shape() -> int:
    """Determine the shape of the label based on the problem type and label mappings."""
    problem_type = config['Experiment'].get('PROBLEM_TYPE', None)
    mappings = config['Labels'].get('MAPPINGS', None)
    if problem_type == 'Multi-Label':
        return sum(len(v) for v in mappings.values())
    elif problem_type in ['Multi-Class', 'Binary']:
        return len(mappings.get(next(iter(mappings))))
    else:
        raise ValueError(f"Invalid PROBLEM_TYPE: {problem_type}")

def preprocess_wrapper(file_path, label, augment: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    """Wrapper function for preprocessing."""
    image, label = tf.py_function(
        func=lambda file_path, label, augment: preprocess_image(file_path, label, augment),
        inp=[file_path, label, augment], 
        Tout=[tf.float32, tf.int32]
    )
    image.set_shape([config['Model']['IMG_SIZE'], config['Model']['IMG_SIZE'], 3])
    label.set_shape([determine_label_shape()])
    return image, label

def apply_performance_settings(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply performance settings to a dataset."""
    AUTOTUNE = tf.data.AUTOTUNE
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)

def create_tf_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Main function to create TensorFlow datasets based on the problem type."""
    tf_datasets = {}
    problem_type = config['Experiment'].get('PROBLEM_TYPE')

    if problem_type == 'Multi-Label':
        print("[INFO] Problem type detected as Multi-Label.")
        for split, df in datasets.items():
            if df is not None:
                tf_datasets[split] = create_tf_dataset(df, ('ImageFile', 'Multi_Labels_Binarized'), split == 'train')
    else:
        print("[INFO] Problem type detected as Multi-Class/Binary.")
        for label in ['Focus_Label', 'StigX_Label', 'StigY_Label']:
            label_datasets = {
                split: create_tf_dataset(
                    df, ('ImageFile', label), split == 'train'
                )
                for split, df in datasets.items()
                if df is not None
            }
            tf_datasets[label] = label_datasets

    for key, ds in tf_datasets.items():
        if isinstance(ds, dict):
            for split in ds:
                ds[split] = apply_performance_settings(ds[split])
        else:
            tf_datasets[key] = apply_performance_settings(ds)

    print("---> TF Datasets created.")
    return tf_datasets


# Global config object for easier parameter management
config = {
    'Model': {
        'IMG_SIZE': 224  # default image size
    },
    'Augmentation': {
        'rotation_factor': 0.2,
        'height_factor': 0.2,
        'width_factor': 0.2,
        'contrast_factor': 0.2
    }
}

def create_preprocessing_layers() -> keras.Sequential:
    """Create preprocessing layers for resizing and rescaling images."""
    img_size = config['Model']['IMG_SIZE']
    return keras.Sequential([
        layers.Resizing(img_size, img_size),
        layers.Rescaling(1./255)
    ])

def create_augmentation_layers() -> keras.Sequential:
    """Create data augmentation layers."""
    aug_config = config['Augmentation']
    try:
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(aug_config['rotation_factor']),
            layers.RandomTranslation(
                height_factor=aug_config['height_factor'],
                width_factor=aug_config['width_factor'],
                fill_mode="reflect"
            ),
            layers.RandomContrast(aug_config['contrast_factor']),
        ])
    except Exception as e:
        print(f"An error occurred while creating augmentation layers: {e}")
        return None

def read_and_convert_image(file_path: str) -> tf.Tensor:
    """Read an image from a file and convert it to a 3-channel tensor."""
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Failed to read the image.")
        return None
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)
    return tf.image.grayscale_to_rgb(image)

def handle_tensor_input(file_path_or_tensor: Union[str, tf.Tensor]) -> tf.Tensor:
    """Handle different types of input and return a 3-channel tensor image."""
    if not isinstance(file_path_or_tensor, tf.Tensor):
        return read_and_convert_image(file_path_or_tensor)
    if len(file_path_or_tensor.shape) == 4:
        return file_path_or_tensor
    else:
        return read_and_convert_image(file_path_or_tensor.numpy().decode('utf-8'))

def preprocess_image(file_path_or_tensor, label, augment: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    try:
        image = handle_tensor_input(file_path_or_tensor)
        preprocess_seq = create_preprocessing_layers()
        image = preprocess_seq(image)
        
        if augment:
            augment_seq = create_augmentation_layers()
            if augment_seq is not None:
                image = augment_seq(image)
                image = tf.clip_by_value(image, 0, 1)
        
        return image, label
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise
# Main function to integrate all steps


# Configure for dataset creation
csv_config = {
    'CSV': {
        'COLUMNS_TO_READ': ['ImageFile', 'Focus_Offset (V)', 'Stig_Offset_X (V)', 'Stig_Offset_Y (V)']
    },
    'Thresholds': {
        'FOCUS_LOW': 30,  # Lower focus threshold
        'FOCUS_HIGH': 60,  # Upper focus threshold
        'STIGX_LOW': 1,  # Lower astigmatism threshold
        'STIGX_HIGH': 2,  # Upper astigmatism threshold
        'STIGY_LOW': 1,  # Lower astigmatism threshold
        'STIGY_HIGH': 2,  # Upper astigmatism threshold
    },
    'Paths': {  # Data and model paths
        'DATA_FILE': "combined_output.csv",
        'OLD_BASE_PATH': "D:\\DOE\\",
        # 'NEW_BASE_PATH': "Y:\\User\\Aaron-HX38\\DOE\\",
        # 'NEW_BASE_PATH': "C:\\Users\\aaron.woods\\OneDrive - Thermo Fisher Scientific\\Documents\\GitHub\\Image-Classification\\",
        'NEW_BASE_PATH': "C:\\Users\\aaron.woods\\OneDrive - Thermo Fisher Scientific\\Desktop\\Dec 24\\",
    },
    'SAMPLE_FRAC': 1.0,  # Fraction of the data to use for quicker prototyping. 1.0 means use all data.
}
config.update(csv_config)
# config['Experiment']['PROBLEM_TYPE'] = 'Multi-Class'
# config['Experiment']['PROBLEM_TYPE'] = 'Multi-Label'



# Main function to integrate all steps
def main_pipeline(config: Dict):
    print("===== Preprocessing CSV Data =====")
    # data = read_csv(config, clean=True)
    data = read_csv(config)
    data = update_image_paths(data)
    data = clean_csv(data)
    data, label_encoders, mlb_classess  = generate_labels(data)
    data = shuffle_and_reset_index(data)
    print("===== Preparing TensorFlow Datasets =====")
    datasets = prepare_datasets(data)
    info = compute_and_store_class_weights(datasets, label_encoders)
    datasets = create_tf_datasets(datasets)
    print("===== Preprocessing Complete =====")
    return datasets, info

datasets, info = main_pipeline(config)
