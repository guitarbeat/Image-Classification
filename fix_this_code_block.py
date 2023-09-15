# Creating TensorFlow datasets

def create_tf_dataset(df: pd.DataFrame, columns: Tuple[str, str], is_training: bool) -> tf.data.Dataset:
    """Create a TensorFlow dataset from a DataFrame."""
    ds = tf.data.Dataset.from_tensor_slices((df[columns[0]].values, df[columns[1]].values))
    return preprocess_single_dataset(ds, is_training)

def preprocess_single_dataset(ds, is_training: bool = False) -> tf.data.Dataset:
    """Apply preprocessing to a single dataset."""
    ds = ds.map(lambda file_path, label: preprocess_wrapper(file_path, label, is_training))
    try:
        for file_path, label in ds.take(1):
            tf.print("[DEBUG] preprocess_single_dataset: shape and dtype:", file_path.shape, file_path.dtype, label.shape, label.dtype)
    except Exception as e:
        print(f"[ERROR] Exception caught in preprocess_single_dataset: {e}")
    return ds

def determine_label_shape() -> int:
    """Determine the shape of the label based on the problem type and label mappings."""
    problem_type = config['Experiment'].get('PROBLEM_TYPE', None)
    mappings = config['Labels'].get('MAPPINGS', None)
    if problem_type == 'Multi-Label':
        label_shape = sum(len(v) for v in mappings.values())  # for the Multi-Label case
        print(f"[DEBUG] determine_label_shape - Multi-Label label shape: {label_shape}")
        return sum(len(v) for v in mappings.values())
    if problem_type in ['Multi-Class', 'Binary']:
        label_shape = len(mappings.get(next(iter(mappings))))  # for the Multi-Class/Binary case
        print(f"[DEBUG] determine_label_shape - Multi-Class/Binary label shape: {label_shape}")
        return len(mappings.get(next(iter(mappings))))
    raise ValueError(f"Invalid PROBLEM_TYPE: {problem_type}")

def preprocess_wrapper(file_path, label, augment: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    """Wrapper function for preprocessing."""
    try:
        image, label = tf.py_function(
            func=lambda file_path, label, augment: preprocess_image(file_path, label, augment),
            inp=[file_path, label, augment], 
            Tout=[tf.float32, tf.int32] # Expects both the image and label to be in these data types
        )
        image.set_shape([config['Model']['IMG_SIZE'], config['Model']['IMG_SIZE'], 3])
        label.set_shape([determine_label_shape()])
        
        tf.print("[DEBUG] preprocess_wrapper: shape and dtype of image and label:", image.shape, image.dtype, label.shape, label.dtype)
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


def create_multi_label_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    print("[INFO] Problem type detected as Multi-Label.")
    tf_datasets = {
        split: create_tf_dataset(df, ('ImageFile', 'Multi_Labels_Binarized'), split == 'train')
        for split, df in datasets.items()
        if df is not None
    }
    print("---> Multi-Label TF Datasets created.")
    return {k: apply_performance_settings(v) for k, v in tf_datasets.items()}

def create_multi_class_or_binary_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    print("[INFO] Problem type detected as Multi-Class/Binary.")
    tf_datasets = {}
    for label in ['Focus_Label', 'StigX_Label', 'StigY_Label']:
        label_datasets = {
            split: create_tf_dataset(df, ('ImageFile', label), split == 'train')
            for split, df in datasets.items()
            if df is not None
        }
        tf_datasets[label] = {k: apply_performance_settings(v) for k, v in label_datasets.items()}
    print("---> Multi-Class/Binary TF Datasets created.")
    return tf_datasets

def create_tf_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Main function to create TensorFlow datasets based on the problem type."""
    problem_type = config['Experiment'].get('PROBLEM_TYPE')
    
    if problem_type == 'Multi-Label':
        return create_multi_label_datasets(datasets)
    
    return create_multi_class_or_binary_datasets(datasets)





# Image Augmentation and Preprocessing

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
    
def handle_tensor_input(file_path_or_tensor: Union[str, tf.Tensor]) -> tf.Tensor:
    """Handle different types of input and return a 3-channel tensor image."""
    if not isinstance(file_path_or_tensor, tf.Tensor):
        return read_and_convert_image(file_path_or_tensor)
    if len(file_path_or_tensor.shape) == 4:
        return file_path_or_tensor
    else:
        return read_and_convert_image(file_path_or_tensor.numpy().decode('utf-8'))
    
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

