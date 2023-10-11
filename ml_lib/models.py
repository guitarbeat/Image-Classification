# models.py

### Model Building (Define the Model)

def add_multi_output_heads(base_layer, num_classes: int, output_names: List[str]) -> List[keras.layers.Layer]:
    """Creates multiple output heads for a given base layer."""
    outputs = []
    for i in range(num_classes):
        x = layers.Dense(128, activation="relu")(base_layer)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(3, activation="softmax", name=output_names[i])(x)  # Naming each output layer
        outputs.append(x)
    return outputs

def determine_activation_and_units(num_classes: int) -> Tuple[List[str], List[int]]:
    """Determines the activation functions and units based on the number of classes and config settings."""
    problem_type = config.get('Experiment').get('PROBLEM_TYPE')
    if problem_type in ['Multi-Label', 'Binary', 'Multi-Class', 'Multi-Output']:
        return {
            'Multi-Label': (["sigmoid"] * num_classes, [1] * num_classes),
            'Binary': (["sigmoid"], [1]),
            'Multi-Class': (["softmax"], [num_classes]),
            'Multi-Output': (["softmax"] * num_classes, [3] * num_classes)  # Assuming each output has 3 classes
        }[problem_type]
    raise ValueError(f"Invalid problem_type: {problem_type}")

def create_transfer_model(base_model, input_shape: Tuple[int, int, int], num_classes: int, hidden_units: List[int], dropout_rate: float, regularizer_rate: float, output_names: List[str] = None) -> keras.Model:
    """Creates a transfer learning model based on the provided base model."""
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D()
    ])
    for units in hidden_units:
        model.add(layers.Dense(units, kernel_regularizer=keras.regularizers.l2(regularizer_rate), bias_regularizer=keras.regularizers.l2(regularizer_rate)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dropout_rate))
    
    activations, units_list = determine_activation_and_units(num_classes)
    if len(activations) == 1:
        model.add(layers.Dense(units_list[0], activation=activations[0]))
        return model
    
    # output_names = output_names or list(config['Labels']['MAPPINGS'].keys())
    output_names = list(config['Labels']['MAPPINGS'].keys())

    outputs = add_multi_output_heads(model.layers[-1].output, num_classes, output_names)
    return keras.Model(inputs=model.input, outputs=outputs)

def create_specific_transfer_model(base_model_class, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """Helper function to create specific transfer models."""
    base_model = base_model_class(input_shape=input_shape, include_top=False, weights='imagenet')
    return create_transfer_model(base_model, input_shape, num_classes, [128, 64], 0.5, 0.001, output_names=config['Labels']['MAPPINGS'].keys())

def create_mobilenetv2_transfer_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    return create_specific_transfer_model(tf.keras.applications.MobileNetV2, input_shape, num_classes)

def create_inceptionv3_transfer_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    return create_specific_transfer_model(tf.keras.applications.InceptionV3, input_shape, num_classes)

def create_resnet50_transfer_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    return create_specific_transfer_model(tf.keras.applications.ResNet50, input_shape, num_classes)

# Define the function to create a basic CNN model
def create_basic_cnn_model(input_shape, num_classes):
    conv2d_filter_size = (3, 3)
    conv2d_activation = 'relu'
    dense_activation = 'relu'
    num_conv_blocks = 3

    inputs = keras.Input(shape=input_shape)

    x = inputs

    for _ in range(num_conv_blocks):
        x = layers.Conv2D(32 * (2**_), conv2d_filter_size, activation=conv2d_activation, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=dense_activation)(x)

    activations, units_list = determine_activation_and_units(num_classes)
    if len(activations) == 1:
        # Single output
        x = layers.Dense(units_list[0], activation=activations[0])(x)
        return keras.Model(inputs=inputs, outputs=x)
    else:
        # Multiple outputs
        outputs = add_multi_output_heads(x, num_classes, output_names=list(config['Labels']['MAPPINGS'].keys()))
        return keras.Model(inputs=inputs, outputs=outputs)

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

    activations, units_list = determine_activation_and_units(num_classes)
    if len(activations) == 1:
        # Single output
        x = layers.Dense(units_list[0], activation=activations[0])(x)
        return keras.Model(inputs=inputs, outputs=x)
    else:
        # Multiple outputs
        outputs = add_multi_output_heads(x, num_classes, output_names=list(config['Labels']['MAPPINGS'].keys()))
        return keras.Model(inputs=inputs, outputs=outputs)

# Model Selection function to select which model to use
def select_model(model_name: str, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
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

