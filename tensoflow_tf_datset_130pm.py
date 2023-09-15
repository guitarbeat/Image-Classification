# Creating TensorFlow datasets

def tf_debugging(tensor_or_dataset, message: str = ""):
    """
    Debugging function for TensorFlow tensors or datasets.
    
    Parameters:
    - tensor_or_dataset: TensorFlow tensor or dataset to inspect.
    - message: Optional message to display alongside debugging info.
    """
    if isinstance(tensor_or_dataset, tf.data.Dataset):
        for data, label in tensor_or_dataset.take(1):
            tf.print(f"{message} Shape and dtype of data: {data.shape} {data.dtype}, Shape and dtype of label: {label.shape} {label.dtype}")
    else:
        tf.print(f"{message} Shape and dtype: {tensor_or_dataset.shape} {tensor_or_dataset.dtype}")

def create_tf_dataset(df: pd.DataFrame, columns: tuple, is_training: bool, label_shape: int) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from a DataFrame.
    
    Parameters:
    - df: DataFrame containing the data.
    - columns: Tuple containing the name of the image file column and label column.
    - is_training: Whether the dataset is for training.
    - label_shape: The shape of the label tensor.
    
    Returns:
    - TensorFlow dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices((df[columns[0]].values, df[columns[1]].values))
    return preprocess_single_dataset(ds, is_training, label_shape)

def preprocess_single_dataset(ds, is_training: bool = False, label_shape: int = 1) -> tf.data.Dataset:
    """
    Apply preprocessing to a single dataset.
    
    Parameters:
    - ds: TensorFlow dataset to preprocess.
    - is_training: Whether the dataset is for training.
    - label_shape: The shape of the label tensor.
    
    Returns:
    - Preprocessed TensorFlow dataset.
    """
    try:
        ds = ds.map(lambda file_path, label: preprocess_wrapper(file_path, label, is_training, label_shape))
        tf_debugging(ds, "preprocess_single_dataset - Dataset:")
    except Exception as e:
        print(f"[ERROR] Exception caught in preprocess_single_dataset: {e}")
    return ds

def preprocess_wrapper(file_path, label, augment: bool, label_shape: int):
    """
    Wrapper function for preprocessing.
    
    Parameters:
    - file_path: File path of the image.
    - label: Label of the image.
    - augment: Whether to apply augmentation.
    - label_shape: The shape of the label tensor.
    
    Returns:
    - Preprocessed image and label.
    """
    try:
        image, label = tf.py_function(
            func=lambda file_path, label, augment: preprocess_image(file_path, label, augment),
            inp=[file_path, label, augment],
            Tout=[tf.float32, tf.int32]
        )
        image.set_shape([config['Model']['IMG_SIZE'], config['Model']['IMG_SIZE'], 3])
        label.set_shape([label_shape])
        tf_debugging(image, "preprocess_wrapper - Image:")
        tf_debugging(label, "preprocess_wrapper - Label:")
    except Exception as e:
        print(f"[ERROR] Exception caught in preprocess_wrapper: {e}")
    return image, label

def apply_performance_settings(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply performance settings to a dataset."""
    try:
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        
        for data, label in dataset.take(1):
            tf.print("[DEBUG] apply_performance_settings: shape and dtype of data and label:", data.shape, data.dtype, label.shape, label.dtype)
    except Exception as e:
        print(f"[ERROR] Exception caught in apply_performance_settings: {e}")
    return dataset

def create_multi_class_or_binary_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Create multi-class or binary datasets.
    
    Parameters:
    - datasets: Dictionary containing DataFrames for train, validation, and test splits.
    
    Returns:
    - Dictionary containing TensorFlow datasets for train, validation, and test splits for each label.
    """
    tf_datasets = {}
    label_shape = 1  # For multi-class or binary datasets, label shape is [1]
    for label in ['Focus_Label', 'StigX_Label', 'StigY_Label']:
        label_datasets = {
            split: create_tf_dataset(df, ('ImageFile', label), split == 'train', label_shape)
            for split, df in datasets.items()
        }
        # Setting label shape for multi-class/binary datasets
        for ds in label_datasets.values():
            ds = ds.map(lambda image, label: (image, tf.reshape(label, [1])))
        tf_datasets[label] = {k: apply_performance_settings(v) for k, v in label_datasets.items()}
    return tf_datasets

def create_multi_label_datasets(datasets: Dict[str, pd.DataFrame], num_labels: int) -> Dict[str, Any]:
    """
    Create multi-label datasets.
    
    Parameters:
    - datasets: Dictionary containing DataFrames for train, validation, and test splits.
    - num_labels: Number of possible labels for multi-label classification.
    
    Returns:
    - Dictionary containing TensorFlow datasets for train, validation, and test splits for each label.
    """
    tf_datasets = {}
    label_shape = num_labels  # For multi-label datasets, label shape is based on the number of possible labels
    for label in ['Focus_Label', 'StigX_Label', 'StigY_Label']:
        label_datasets = {
            split: create_tf_dataset(df, ('ImageFile', label), split == 'train', label_shape)
            for split, df in datasets.items()
        }
        # Setting label shape for multi-label datasets
        for ds in label_datasets.values():
            ds = ds.map(lambda image, label: (image, tf.reshape(label, [num_labels])))
        tf_datasets[label] = {k: apply_performance_settings(v) for k, v in label_datasets.items()}
    return tf_datasets

def create_tf_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Create TensorFlow datasets based on the problem type specified in the config.
    
    Parameters:
    - datasets: Dictionary containing DataFrames for train, validation, and test splits.
    - config: Configuration dictionary containing experiment settings and parameters.
    
    Returns:
    - Dictionary containing TensorFlow datasets for train, validation, and test splits for each label.
    """
    problem_type = config['Experiment']['PROBLEM_TYPE']

    if problem_type in ['Multi-Class', 'Binary']:
        return create_multi_class_or_binary_datasets(datasets)
    elif problem_type == 'Multi-Label':
        # Determine the number of possible labels for one of the label types (assuming all have the same number of labels)
        num_labels = len(next(iter(config['Labels']['MAPPINGS'].values())))
        return create_multi_label_datasets(datasets, num_labels)
    else:
        raise ValueError("Invalid problem_type in config. Choose from 'Multi-Class', 'Binary', 'Multi-Label'.")

