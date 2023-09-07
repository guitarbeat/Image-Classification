from collections import defaultdict
from typing import List, Dict, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Utility Functions

def compute_weights_and_info(labels: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[int, Dict[str, int], Dict[str, float]]]]:
    """Compute class weights and information about the dataset."""
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=np.array(labels))
    info = {
        'Total': len(labels),
        'Counts': dict(zip(unique_labels, counts)),
        'Weights': dict(zip(unique_labels, np.round(weights, 2)))
    }
    return weights, unique_labels, info

# Data Loading

def load_data_from_dataframe_multi_labels(data_df: pd.DataFrame, label_columns: List[str]) -> Dict[str, Dict[str, int]]:
    """Load multiple labels from a DataFrame."""
    
    labels_by_file_path = {
        row['ImageFile']: {label_column: row[label_column] for label_column in label_columns}
        for _, row in data_df.iterrows()
    }
    return labels_by_file_path

# Data Splitting

def split_and_stratify(file_paths: List[str], labels: List[str], train_size: float, val_size: float) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Split and stratify dataset into training, validation, and test sets."""
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, train_size=train_size, stratify=labels)
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(temp_paths, temp_labels, train_size=val_size, stratify=temp_labels)
    return train_paths, valid_paths, test_paths, train_labels, valid_labels, test_labels



def map_labels_to_integers(labels: List[str], label_column: str, label_to_int_mappings: Dict[str, int]) -> List[int]:
    """Map a list of string labels to their corresponding integer values based on the provided mapping."""
    return [label_to_int_mappings[label_column].get(label, -1) for label in labels]

def create_tf_dataset(file_paths: List[str], labels: List[int]) -> tf.data.Dataset:
    """Create a TensorFlow dataset from file paths and labels."""
    return tf.data.Dataset.from_tensor_slices((file_paths, labels))
