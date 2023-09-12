# V7 Function for Preparing CSV
# Data Preprocessing and Dataset Preparation

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.data import Dataset
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def preprocess_and_prepare_tf_datasets(config: dict) -> dict:
    """Preprocess a CSV data file and prepare TensorFlow datasets."""
    
    print("===== Preprocessing CSV Data =====")
    
    # Functionality to read the data
    data_file_path = os.path.join(config['Paths']['BASE_DIR'], config['Paths']['DATA_FILE'])
    print(f"---> Reading data from: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Error: File does not exist - {data_file_path}")
    
    try:
        data = pd.read_csv(data_file_path)
        print("---> Data read successfully.")
    except Exception as e:
        raise ValueError(f"Error: Could not read data - {e}") from e
    
    # Functionality to update image paths
    print("---> Updating image file paths...")
    data['ImageFile'] = data['ImageFile'].str.replace(config['Paths']['OLD_BASE_PATH'], config['Paths']['NEW_BASE_PATH'], regex=False)
    print("---> Image paths updated.")
    
    # Functionality to generate labels for Focus, StigX, and StigY
    print("---> Generating labels for Focus, StigX, and StigY...")
    focus_conditions = [
        (data['Focus_Offset (V)'].abs() <= config['Thresholds']['FOCUS_LOW']),
        (data['Focus_Offset (V)'].abs() <= config['Thresholds']['FOCUS_HIGH']),
        (data['Focus_Offset (V)'].abs() > config['Thresholds']['FOCUS_HIGH'])]
    stig_x_conditions = [
        (data['Stig_Offset_X (V)'].abs() <= config['Thresholds']['STIG_LOW']),
        (data['Stig_Offset_X (V)'].abs() <= config['Thresholds']['STIG_HIGH']),
        (data['Stig_Offset_X (V)'].abs() > config['Thresholds']['STIG_HIGH'])]
    stig_y_conditions = [
        (data['Stig_Offset_Y (V)'].abs() <= config['Thresholds']['STIG_LOW']),
        (data['Stig_Offset_Y (V)'].abs() <= config['Thresholds']['STIG_HIGH']),
        (data['Stig_Offset_Y (V)'].abs() > config['Thresholds']['STIG_HIGH'])]
    focus_choices = list(config['Labels']['MAPPINGS']['Focus_Label'].keys())
    stig_x_choices = list(config['Labels']['MAPPINGS']['StigX_Label'].keys())
    stig_y_choices = list(config['Labels']['MAPPINGS']['StigY_Label'].keys())
    data['Focus_Label'] = np.select(focus_conditions, focus_choices)
    data['StigX_Label'] = np.select(stig_x_conditions, stig_x_choices)
    data['StigY_Label'] = np.select(stig_y_conditions, stig_y_choices)
    print("---> Labels generated.")

    # # Generate One-Hot Encoded Labels for Focus, StigX, and StigY
    # print("---> Generating one-hot encoded labels for Focus, StigX, and StigY...")
    # # One-hot encode the labels
    # focus_one_hot = pd.get_dummies(data['Focus_Label'], prefix='Focus')
    # stig_x_one_hot = pd.get_dummies(data['StigX_Label'], prefix='StigX')
    # stig_y_one_hot = pd.get_dummies(data['StigY_Label'], prefix='StigY')
    # # Concatenate one-hot columns to the dataframe
    # data = pd.concat([data, focus_one_hot, stig_x_one_hot, stig_y_one_hot], axis=1)
    # print("---> One-hot encoded labels generated.")
    
    print("===== Preparing TensorFlow Datasets =====")
    
    labeled_df = data
    datasets = {}
    labels = list(config['Labels']['MAPPINGS'].keys())

    # Shuffle and Reset Index
    print("---> Shuffling and resetting index...")
    labeled_df = labeled_df.sample(frac=1, random_state=config['Experiment']['RANDOM_SEED']).reset_index(drop=True)
    print("---> Data shuffled and index reset.")
    
    for label in labels:
        
        print(f"---> Preparing datasets for label: {label}")
        datasets[label] = {'train': None, 'valid': None, 'test': None, 'info': {}}
        
        # Split Data
        train_df, temp_df = train_test_split(labeled_df, test_size=1 - config['Model']['TRAIN_SIZE'], random_state=config['Experiment']['RANDOM_SEED'])
        val_df, test_df = train_test_split(temp_df, test_size=1 - config['Model']['VAL_SIZE'], random_state=config['Experiment']['RANDOM_SEED'])

        # Create TensorFlow Datasets (without one-hot encoded labels)
        train_ds = Dataset.from_tensor_slices((train_df['ImageFile'].values, train_df[label].map(config['Labels']['MAPPINGS'][label]).values))
        val_ds = Dataset.from_tensor_slices((val_df['ImageFile'].values, val_df[label].map(config['Labels']['MAPPINGS'][label]).values))
        test_ds = Dataset.from_tensor_slices((test_df['ImageFile'].values, test_df[label].map(config['Labels']['MAPPINGS'][label]).values))

        # one_hot_columns = [col for col in data.columns if label in col]  # Find the corresponding one-hot columns
        # # Create TensorFlow Datasets with one-hot encoded labels
        # train_ds = Dataset.from_tensor_slices((train_df['ImageFile'].values, train_df[one_hot_columns].values))
        # val_ds = Dataset.from_tensor_slices((val_df['ImageFile'].values, val_df[one_hot_columns].values))
        # test_ds = Dataset.from_tensor_slices((test_df['ImageFile'].values, test_df[one_hot_columns].values))

        # Apply Preprocessing using your preprocessing functions
        train_ds, val_ds, test_ds = preprocess_images(train_ds, val_ds, test_ds, config)
        
        # # Debug/Check Statement
        # print("---> Debug Check: Verifying the first element of the train_ds after preprocessing")
        # for img, lbl in train_ds.take(1):
        #     print(f"Image shape: {img.shape}, Label: {lbl.numpy()}")


        # Configure for Performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Store Datasets
        datasets[label]['train'] = train_ds
        datasets[label]['valid'] = val_ds
        datasets[label]['test'] = test_ds

        # Compute and Store Class Weights and Info for each split
        for split, df in zip(['Training', 'Validation', 'Test'], [train_df, val_df, test_df]):
            unique_labels = df[label].unique()
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=df[label])
            class_weights_dict = dict(zip(unique_labels, class_weights))

            datasets[label]['info'][split] = {
                'Total': len(df),
                'ClassInfo': {cls: {'Count': cnt, 'Weight': class_weights_dict[cls]} for cls, cnt in Counter(df[label]).items()}
            }
        print(f"---> Datasets prepared for label: {label}")

    print("===== Preprocessing and Dataset Preparation Complete =====")
   

    return datasets

# Preprocess and prepare datasets
datasets = preprocess_and_prepare_tf_datasets(config)

# V7.5 Function for Preparing CSV
# Data Preprocessing and Dataset Preparation

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.data import Dataset
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_sample_weight


def preprocess_and_prepare_tf_datasets(config: dict) -> dict:
    """Preprocess a CSV data file and prepare TensorFlow datasets."""
    
    print("===== Preprocessing CSV Data =====")
    
    # Functionality to read the data
    data_file_path = os.path.join(config['Paths']['BASE_DIR'], config['Paths']['DATA_FILE'])
    print(f"---> Reading data from: {data_file_path}")
    
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Error: File does not exist - {data_file_path}")
    
    try:
        data = pd.read_csv(data_file_path)
        print("---> Data read successfully.")
    except Exception as e:
        raise ValueError(f"Error: Could not read data - {e}") from e
    
    # Functionality to update image paths
    print("---> Updating image file paths...")
    data['ImageFile'] = data['ImageFile'].str.replace(config['Paths']['OLD_BASE_PATH'], config['Paths']['NEW_BASE_PATH'], regex=False)
    print("---> Image paths updated.")
    # === New Code Starts ===
    # Create a DataFrame for storing offset values
    offset_values_df = data[['ImageFile', 'Focus_Offset (V)', 'Stig_Offset_X (V)', 'Stig_Offset_Y (V)']].copy()
    # === New Code Ends ===
    # Functionality to generate labels for Focus, StigX, and StigY
    print("---> Generating labels for Focus, StigX, and StigY...")
    focus_conditions = [
        (data['Focus_Offset (V)'].abs() <= config['Thresholds']['FOCUS_LOW']),
        (data['Focus_Offset (V)'].abs() <= config['Thresholds']['FOCUS_HIGH']),
        (data['Focus_Offset (V)'].abs() > config['Thresholds']['FOCUS_HIGH'])]
    stig_x_conditions = [
        (data['Stig_Offset_X (V)'].abs() <= config['Thresholds']['STIG_LOW']),
        (data['Stig_Offset_X (V)'].abs() <= config['Thresholds']['STIG_HIGH']),
        (data['Stig_Offset_X (V)'].abs() > config['Thresholds']['STIG_HIGH'])]
    stig_y_conditions = [
        (data['Stig_Offset_Y (V)'].abs() <= config['Thresholds']['STIG_LOW']),
        (data['Stig_Offset_Y (V)'].abs() <= config['Thresholds']['STIG_HIGH']),
        (data['Stig_Offset_Y (V)'].abs() > config['Thresholds']['STIG_HIGH'])]
    focus_choices = list(config['Labels']['MAPPINGS']['Focus_Label'].keys())
    stig_x_choices = list(config['Labels']['MAPPINGS']['StigX_Label'].keys())
    stig_y_choices = list(config['Labels']['MAPPINGS']['StigY_Label'].keys())
    data['Focus_Label'] = np.select(focus_conditions, focus_choices)
    data['StigX_Label'] = np.select(stig_x_conditions, stig_x_choices)
    data['StigY_Label'] = np.select(stig_y_conditions, stig_y_choices)
    print("---> Labels generated.")
    data['Multi_Labels'] = data.apply(lambda row: [row['Focus_Label'], row['StigX_Label'], row['StigY_Label']], axis=1)
    print("---> Multi-labels generated.")
    
    print("===== Preparing TensorFlow Datasets =====")
    
    # Multi-label binarization
    mlb = MultiLabelBinarizer()
    data['Multi_Labels_Binarized'] = list(mlb.fit_transform(data['Multi_Labels']))

    # Shuffle and Reset Index
    print("---> Shuffling and resetting index...")
    labeled_df = data.sample(frac=1, random_state=config['Experiment']['RANDOM_SEED']).reset_index(drop=True)
    print("---> Data shuffled and index reset.")
    
    # Split Data
    train_df, temp_df = train_test_split(labeled_df, test_size=1 - config['Model']['TRAIN_SIZE'], random_state=config['Experiment']['RANDOM_SEED'])
    val_df, test_df = train_test_split(temp_df, test_size=1 - config['Model']['VAL_SIZE'], random_state=config['Experiment']['RANDOM_SEED'])

    # Create an empty dictionary to store datasets and info
    datasets = {'train': None, 'valid': None, 'test': None, 'info': {'Training': {}, 'Validation': {}, 'Test': {}}}

    # Create TensorFlow Datasets
    train_ds = Dataset.from_tensor_slices((train_df['ImageFile'].values, list(train_df['Multi_Labels_Binarized'])))
    val_ds = Dataset.from_tensor_slices((val_df['ImageFile'].values, list(val_df['Multi_Labels_Binarized'])))
    test_ds = Dataset.from_tensor_slices((test_df['ImageFile'].values, list(test_df['Multi_Labels_Binarized'])))
    
 
    # Apply Preprocessing using your preprocessing functions
    train_ds, val_ds, test_ds = preprocess_images(train_ds, val_ds, test_ds, config)


    # Store Datasets
    datasets['train'] = train_ds
    datasets['valid'] = val_ds
    datasets['test'] = test_ds

    # Compute and Store Class Weights and Info for each split
    for label in range(len(mlb.classes_)):
        for split, df in zip(['Training', 'Validation', 'Test'], [train_df, val_df, test_df]):
            # Extract the column for the current label from the multi-label binarized labels
            label_column = df['Multi_Labels_Binarized'].apply(lambda x: x[label])

            # Compute class weights
            unique_labels = np.unique(label_column)  # In the binarized label, each label can be either 0 or 1
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=label_column)
            class_weights_dict = dict(zip(unique_labels, class_weights))
            
            # Store the class weights and other info
            datasets['info'][split][mlb.classes_[label]] = {
                'Total': len(df),
                'ClassInfo': {cls: {'Count': cnt, 'Weight': class_weights_dict.get(cls, 0)} for cls, cnt in Counter(label_column).items()}
            }
    # === New Code Starts ===
    # Store the offset values DataFrame in the datasets dictionary
    datasets['offset_values'] = offset_values_df
    # === New Code Ends ===

    print("===== Preprocessing and Dataset Preparation Complete =====")
    
    return datasets


# Preprocess and prepare datasets
dataset = preprocess_and_prepare_tf_datasets(config)
