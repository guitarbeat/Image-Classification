# Standard Libraries
import os

# Third-Party Libraries
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import InceptionV3, ResNet50

# Type Annotations
from typing import List, Dict, Tuple, Union, Any, Optional

# Check for GPU support
if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. Neural nets can be very slow without a GPU.")

# Defining the Models------------------------------------------------------------#

def create_transfer_model(base_model, input_shape: tuple, num_classes: int, hidden_units: list, dropout_rate: float, regularizer_rate: float) -> keras.Model:
    """Creates a transfer learning model based on a given base model."""
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D()
    ])

    for units in hidden_units:
        model.add(layers.Dense(units, kernel_regularizer=keras.regularizers.l2(regularizer_rate), bias_regularizer=keras.regularizers.l2(regularizer_rate)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))

    activation, units = ("sigmoid", 1) if num_classes == 2 else ("softmax", num_classes)
    model.add(layers.Dense(units, activation=activation))

    return model

def create_mobilenetv2_transfer_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Creates a MobileNetV2 based transfer learning model."""
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    return create_transfer_model(base_model, input_shape, num_classes, [128, 64], 0.5, 0.001)

def create_inceptionv3_transfer_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Creates an InceptionV3 based transfer learning model."""
    base_model = tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
    return create_transfer_model(base_model, input_shape, num_classes, [128, 64], 0.5, 0.001)

def create_resnet50_transfer_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """Creates a ResNet50 based transfer learning model."""
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    return create_transfer_model(base_model, input_shape, num_classes, [256, 128], 0.5, 0.001)

# Define the function to create a small version of the Xception network
def create_small_xception_model(input_shape, num_classes):
    # Input layer
    inputs = keras.Input(shape=input_shape)

    # Entry block: Initial Convolution and BatchNormalization
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual for later use

    # Middle flow: Stacking Separable Convolution blocks
    for size in [256, 512, 728]:
        # ReLU activation
        x = layers.Activation("relu")(x)
        # Separable Convolution
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # ReLU activation
        x = layers.Activation("relu")(x)
        # Separable Convolution
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Max Pooling
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual from previous block and add it to the current block
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Exit flow: Final Separable Convolution, BatchNormalization, and Global Average Pooling
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Determine activation and units based on the number of classes
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    # Dropout and Dense output layer
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)

    return keras.Model(inputs, outputs)

# Define the function to create a basic CNN model
def create_basic_cnn_model(input_shape, num_classes):
    conv2d_filter_size = (3, 3)
    conv2d_activation = 'relu'
    dense_activation = 'relu'
    num_conv_blocks = 3

    model = tf.keras.models.Sequential()

    # Explicitly define the input shape
    model.add(tf.keras.layers.Input(shape=input_shape))

    for _ in range(num_conv_blocks):
        model.add(tf.keras.layers.Conv2D(32 * (2**_), conv2d_filter_size, activation=conv2d_activation, padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(128, activation=dense_activation))

    # Determine activation and units based on the number of classes
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    model.add(tf.keras.layers.Dense(units, activation=activation))

    return model

# Model Selection function to select which model to use
def select_model(model_name: str, input_shape: tuple, num_classes: int) -> keras.Model:
    """Selects a model to use based on the given model name."""
    model_map = {
        "mobilenetv2": create_mobilenetv2_transfer_model,
        "inceptionv3": create_inceptionv3_transfer_model,
        "resnet50": create_resnet50_transfer_model,
        "small_xception": create_small_xception_model,
        "basic_cnn": create_basic_cnn_model
    }
    if model_name not in model_map:
        raise ValueError("Invalid model name")

    return model_map[model_name](input_shape, num_classes)



# Load and Preprocess the Data---------------------------------------------------#

# Function to read the data
def read_data(file_path):
    print(f"Reading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print("Data read successfully.")
        return data
    except Exception as e:
        raise ValueError(f"Error reading data: {e}") from e

# Function to determine the label of an image
def get_label(row, focus_threshold, stig_threshold):
    focus_label = "InFocus" if abs(row['Focus_Offset (V)']) <= focus_threshold else "OutOfFocus"
    stig_x_label = "NotStiggedX" if abs(row['Stig_Offset_X (V)']) <= stig_threshold else "StiggedX"
    stig_y_label = "NotStiggedY" if abs(row['Stig_Offset_Y (V)']) <= stig_threshold else "StiggedY"
    return focus_label, stig_x_label, stig_y_label

# Function to update the image path
def update_image_path(row, old_base_path, new_base_path):
    image_file = row['ImageFile']
    if isinstance(image_file, str):
        # print(f"Updating image path for string: {image_file}")
        return image_file.replace(old_base_path, new_base_path)
    print(f"Encountered non-string value in 'ImageFile' column: {image_file}")
    return image_file  # or some default value, or raise an exception, etc.

# Main function for preprocessing
def preprocess_dataframe(base_dir, data_file, old_base_path, new_base_path, focus_threshold, stig_threshold):
    data_file_path = base_dir + data_file
    data = read_data(data_file_path)
    
    # Update image paths
    print("Updating image paths...")
    data['ImageFile'] = data.apply(update_image_path, axis=1, args=(old_base_path, new_base_path))
    
    # Generate labels
    print("Generating labels...")
    data['Focus_Label'], data['StigX_Label'], data['StigY_Label'] = zip(*data.apply(get_label, axis=1, args=(focus_threshold, stig_threshold)))
    
    return data

# Image Processing Functions-----------------------------------------------------#


def create_preprocessing_layers(img_width, img_height, rescale_factor):
    """Creates a Sequential model for image preprocessing."""
    return keras.Sequential([
        layers.Resizing(img_width, img_height),
        layers.Rescaling(rescale_factor)
    ])

def create_augmentation_layers(rotation_factor=0.005, height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)):
    """Creates a Sequential model for image augmentation."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(rotation_factor),
        layers.experimental.preprocessing.RandomTranslation(
            height_factor=height_factor,
            width_factor=width_factor,
            fill_mode="reflect"
        )
    ])

def read_and_convert_image(file_path):
    """Reads and converts a grayscale image to RGB."""
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)
    return tf.image.grayscale_to_rgb(image)

def preprocess_image(file_path, label, augment, preprocess_seq, augment_seq):
    """Unified function to preprocess a single image."""
    file_path = file_path.numpy().decode("utf-8")
    
    # Read and convert the image to RGB
    image = read_and_convert_image(file_path)
    
    # Apply preprocessing
    image = preprocess_seq(image)
    
    # Apply augmentation if needed
    if augment:
        image = augment_seq(image)
        
    return image, label

def preprocess_images(train_ds, valid_ds, test_ds, augment_train, augment_valid, augment_test, img_width=224, img_height=224, rescale_factor=1.0 / 255.0):
    """Function to preprocess and optionally augment datasets."""
    # Create preprocessing and augmentation layers
    preprocess_seq = create_preprocessing_layers(img_width, img_height, rescale_factor)
    augment_seq = create_augmentation_layers()
    
    def preprocess_wrapper(file_path, label, augment):
        image, label = tf.py_function(preprocess_image, [file_path, label, augment, preprocess_seq, augment_seq], [tf.float32, tf.int32])
        label.set_shape(())
        return image, label
    
    # Apply preprocessing and augmentation to datasets
    train_ds = train_ds.map(lambda file_path, label: preprocess_wrapper(file_path, label, augment_train))
    valid_ds = valid_ds.map(lambda file_path, label: preprocess_wrapper(file_path, label, augment_valid))
    test_ds = test_ds.map(lambda file_path, label: preprocess_wrapper(file_path, label, augment_test))

    return train_ds, valid_ds, test_ds

# Data Processing Functions------------------------------------------------------#

# Load multiple labels from a DataFrame
def load_data_from_dataframe_multi_labels(data_df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Dict[str, int]]:
    labels_by_file_path = defaultdict(dict)
    for index, row in data_df.iterrows():
        file_path = row['ImageFile']
        labels = {label_column: row[label_column] for label_column in label_columns}
        labels_by_file_path[file_path].update(labels)
    return labels_by_file_path

# Helper Functions
def print_dataset_info(label_column: str, dataset_info: Dict[str, Dict[str, int]]) -> None:
    """Prints formatted dataset information for better readability."""
    print(f"=== {label_column} ===")
    
    for split_name, info in dataset_info[label_column].items():
        print(f"- {split_name}")
        print(f"  - Total Images: {info['Total']}")
        
        counts = ', '.join([f"{label}: {count}" for label, count in info['Counts'].items()])
        weights = ', '.join([f"{label}: {weight:.2f}" for label, weight in info['Weights'].items()])
        
        print(f"  - Counts: {counts}")
        print(f"  - Weights: {weights}")
    
    print('-' * 30)  # Print a separator for better visual distinction

def map_labels_to_integers(label: str, label_column: str) -> int:
    """Map a string label to its corresponding integer value."""
    return config['LABEL_TO_INT_MAPPINGS'][label_column].get(label, -1)

def split_and_stratify(file_paths: List[str], labels: List[str], train_size: float, val_size: float) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Split dataset into training, validation, and test sets."""
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, train_size=train_size, stratify=labels)
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(temp_paths, temp_labels, train_size=val_size, stratify=temp_labels)
    return train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels

def compute_weights_and_info(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float]]:
    """Compute class weights and dataset information."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=np.array(labels))
    info = {
        'Total': len(labels),
        'Counts': dict(zip(unique_labels, counts)),
        'Weights': dict(zip(unique_labels, np.round(weights, 2)))
    }
    return weights, unique_labels, info

# Data Pipeline
def prepare_dataset(base_dir, data_file, old_base_path, new_base_path, 
                    focus_threshold, stig_threshold, label_columns, train_size=0.8, val_size=0.5):
                    
    # Step 1: Preprocess the original CSV data to get a DataFrame
    processed_data = preprocess_dataframe(base_dir, data_file, old_base_path, new_base_path, focus_threshold, stig_threshold)
    
    # Step 2: Load labels and file paths from the DataFrame
    labels_by_file_path = load_data_from_dataframe_multi_labels(processed_data, label_columns)
    file_paths = list(labels_by_file_path.keys())
    datasets, class_weights, dataset_info = {}, {}, {}
    print("Preparing data for different labels:\n")
    for label_column in label_columns:
        labels = [labels_by_file_path[file_path][label_column] for file_path in file_paths]
        train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels = split_and_stratify(file_paths, labels, train_size, val_size)

        # Compute class weights and dataset information
        train_weights, _, train_info = compute_weights_and_info(train_labels)
        valid_weights, _, valid_info = compute_weights_and_info(valid_labels)
        test_weights, _, test_info = compute_weights_and_info(test_labels)
        dataset_info[label_column] = {'Training': train_info, 'Validation': valid_info, 'Test': test_info}
        # print_dataset_info(label_column, dataset_info)

        # Map string labels to integers
        train_labels = list(map(lambda x: map_labels_to_integers(x, label_column), train_labels))
        valid_labels = list(map(lambda x: map_labels_to_integers(x, label_column), valid_labels))
        test_labels = list(map(lambda x: map_labels_to_integers(x, label_column), test_labels))

        # Create and preprocess TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

        train_ds, valid_ds, test_ds = preprocess_images(train_ds, valid_ds, test_ds, augment_train, augment_valid, augment_test)
        datasets[label_column] = {'train': train_ds, 'valid': valid_ds, 'test': test_ds}
        class_weights[label_column] = {'train': train_weights, 'valid': valid_weights, 'test': test_weights}
    return datasets, class_weights, dataset_info
