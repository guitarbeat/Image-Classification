# model_setup.py

### Model Initialization (Compile the Model)

# # Constants & Configurations
LOSS_CONFIG = {
    'Binary': 'binary_crossentropy',
    'Multi-Class': 'categorical_crossentropy',
    'Multi-Output': ['categorical_crossentropy'] * len(config['Labels']['MAPPINGS']),
    'Multi-Label': 'binary_crossentropy'
}

RECOMMENDED_METRICS = {
    'Binary': ['accuracy', 'binary_crossentropy', 'mean_squared_error'],
    'Multi-Class': ['categorical_accuracy', 'categorical_crossentropy', 'mean_squared_error'],
    'Multi-Output': ['categorical_accuracy'] * len(config['Labels']['MAPPINGS']) + 
                    ['categorical_crossentropy'] * len(config['Labels']['MAPPINGS']) + 
                    ['mean_squared_error'] * len(config['Labels']['MAPPINGS']),
    'Multi-Label': ['binary_accuracy', 'binary_crossentropy', 'mean_squared_error']
}

# Helper Functions
def get_accuracy_metric(problem_type: str) -> str:
    """Determine the accuracy metric based on the problem type."""
    return {'Binary': "accuracy", 'Multi-Label': "binary_accuracy"}.get(problem_type, "categorical_accuracy")

def create_directory(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

# Callback Setup Functions
def setup_common_callbacks() -> List[callbacks.Callback]:
    """Set up common callbacks."""
    return [
        callbacks.EarlyStopping(patience=config['Model']['EARLY_STOPPING_PATIENCE'], restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=config['Model']['REDUCE_LR_PATIENCE'], min_lr=config['Model']['MIN_LR'])
    ]

def setup_specific_callbacks(model_name: str, model_dir: str, problem_type: str) -> List[callbacks.Callback]:
    """Set up model-specific callbacks."""
    datetime_str = datetime.now().strftime("%Y%m%d-%I%M%S%p")
    acc_metric = get_accuracy_metric(problem_type)
    checkpoint_path = os.path.join(model_dir, f"saved_model_{datetime_str}_epoch_{{epoch}}_val_loss_{{val_loss:.2f}}_{acc_metric}_{{{{val_{acc_metric}:.2f}}}}.h5")
    return [
        callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
        callbacks.TensorBoard(log_dir=os.path.join(model_dir, "logs", datetime_str))
    ]

def compile_model(model_name: str, input_shape: tuple, num_classes: int, problem_type: str) -> tf.keras.Model:
    """Compile and return a model."""
    model = select_model(model_name, input_shape, num_classes)
    metrics_to_use = list(set(RECOMMENDED_METRICS.get(problem_type, ['accuracy'])))
    loss_to_use = LOSS_CONFIG.get(problem_type, 'categorical_crossentropy')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config['Model']['LEARNING_RATE']), 
        loss=loss_to_use, 
        metrics=metrics_to_use
    )
    # model.summary()
    
    return model

def compile_and_initialize_models() -> Dict[str, Dict[str, tf.keras.Model]]:
    """Main function to compile and initialize models."""
    input_shape = (config['Model']['IMG_SIZE'], config['Model']['IMG_SIZE'], 3)
    num_classes = 3
    problem_type = config['Experiment']['PROBLEM_TYPE']

    experiment_name = config['Experiment']['NAME']
    base_dir = f"./{experiment_name}"
    create_directory(base_dir)

    common_callbacks = setup_common_callbacks()
    label_names = config['Labels']['MAPPINGS'].keys() if problem_type in ['Multi-Class', 'Multi-Output'] else ['']

    compiled_models = {}
    for label_name in label_names:
        label_dir = os.path.join(base_dir, label_name)
        create_directory(label_dir)

        for model_name in ['mobilenetv2', 'inceptionv3', 'resnet50', 'small_xception', 'basic_cnn']:
            model_dir = os.path.join(label_dir, model_name)
            create_directory(model_dir)
            
            specific_callbacks = setup_specific_callbacks(model_name, model_dir, problem_type)
            all_callbacks = common_callbacks + specific_callbacks
            
            model = compile_model(model_name, input_shape, num_classes, problem_type)
            compiled_models[model_name] = {'model': model, 'callbacks': all_callbacks}
    
    return compiled_models

# Execution
compiled_models = compile_and_initialize_models()
print("Models compiled and initialized successfully.")


