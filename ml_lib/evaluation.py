# evaluation.py

# Graph Version Code for Confusion Matrix

from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, encoder=None, figsize=(10, 10), text_size=15, norm=False, savefig=False, output_num=None):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    Args:
    y_true: Array of truth labels.
    y_pred: Array of predicted labels.
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    encoder: Fitted LabelEncoder to transform integer labels back to original string labels.
    ... (other arguments as before)

    Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
    """
    
    # Error handling for array shapes
    if y_true.shape != y_pred.shape:
        print(f"Error: Mismatch in shapes of y_true ({y_true.shape}) and y_pred ({y_pred.shape}). Skipping this output.")
        return
    
    if encoder:
        y_true = encoder.inverse_transform(y_true)
        y_pred = encoder.inverse_transform(y_pred)
    
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    labels = classes if classes else np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = f"{cm[i, j]}" if not norm else f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)"
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, text, horizontalalignment="center", color=color, size=text_size)

    # Save the figure to the current working directory
    if savefig and output_num is not None:
        fig.savefig(f"confusion_matrix_output_{output_num + 1}.png")




def make_confusion_matrix_multi_output(y_true, y_preds, classes=None, encoders=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels for multi-output models.

    Args:
    y_true: 2D array of truth labels for multiple outputs.
    y_preds: 2D array of predicted labels for multiple outputs.
    classes: List of arrays of class labels (e.g. string form) for each output. If `None`, integer labels are used.
    encoders: List of fitted LabelEncoders to transform integer labels back to original string labels for each output.
    ... (other arguments as before)

    Returns:
    A list of labelled confusion matrix plots comparing y_true and y_preds for each output.
    """
    
    # Error handling for array shapes
    if y_true.shape != y_preds.shape:
        print(f"Error: Mismatch in shapes of y_true ({y_true.shape}) and y_preds ({y_preds.shape}). Cannot proceed.")
        return
    
    num_outputs = y_true.shape[1]
    
    for i in range(num_outputs):
        print(f"Confusion Matrix for Output {i + 1}:")
        encoder = encoders[i] if encoders else None
        make_confusion_matrix(y_true[:, i], np.argmax(y_preds[:, i], axis=1), classes[i] if classes else None, encoder, figsize, text_size, norm, savefig, i)
        plt.show()


# Graph Version Code for Confusion Matrix

from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, encoder=None, figsize=(10, 10), text_size=15, norm=False, savefig=False, output_num=None):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    Args:
    y_true: Array of truth labels.
    y_pred: Array of predicted labels.
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    encoder: Fitted LabelEncoder to transform integer labels back to original string labels.
    ... (other arguments as before)

    Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
    """
    
    # Error handling for array shapes
    if y_true.shape != y_pred.shape:
        print(f"Error: Mismatch in shapes of y_true ({y_true.shape}) and y_pred ({y_pred.shape}). Skipping this output.")
        return
    
    if encoder:
        y_true = encoder.inverse_transform(y_true)
        y_pred = encoder.inverse_transform(y_pred)
    
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    labels = classes if classes else np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = f"{cm[i, j]}" if not norm else f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)"
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, text, horizontalalignment="center", color=color, size=text_size)

    # Save the figure to the current working directory
    if savefig and output_num is not None:
        fig.savefig(f"confusion_matrix_output_{output_num + 1}.png")




def make_confusion_matrix_multi_output(y_true, y_preds, classes=None, encoders=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels for multi-output models.

    Args:
    y_true: 2D array of truth labels for multiple outputs.
    y_preds: 2D array of predicted labels for multiple outputs.
    classes: List of arrays of class labels (e.g. string form) for each output. If `None`, integer labels are used.
    encoders: List of fitted LabelEncoders to transform integer labels back to original string labels for each output.
    ... (other arguments as before)

    Returns:
    A list of labelled confusion matrix plots comparing y_true and y_preds for each output.
    """
    
    # Error handling for array shapes
    if y_true.shape != y_preds.shape:
        print(f"Error: Mismatch in shapes of y_true ({y_true.shape}) and y_preds ({y_preds.shape}). Cannot proceed.")
        return
    
    num_outputs = y_true.shape[1]
    
    for i in range(num_outputs):
        print(f"Confusion Matrix for Output {i + 1}:")
        encoder = encoders[i] if encoders else None
        make_confusion_matrix(y_true[:, i], np.argmax(y_preds[:, i], axis=1), classes[i] if classes else None, encoder, figsize, text_size, norm, savefig, i)
        plt.show()


### Function for Confusion Matrix

def make_confusion_matrix_multi_output(y_true, y_preds, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels for multi-output models.

    Args:
    y_true: List of arrays of truth labels (must be same shape as y_preds).
    y_preds: List of arrays of predicted labels (must be same shape as y_true).
    classes: List of arrays of class labels (e.g. string form). If `None`, integer labels are used.
    ... (other arguments as before)

    Returns:
    A list of labelled confusion matrix plots comparing y_true and y_preds for each output.
    """
    # Check if y_true and y_preds are lists, if not, convert them to lists (for single-output compatibility)
    if not isinstance(y_true, list):
        y_true = [y_true]
    if not isinstance(y_preds, list):
        y_preds = [y_preds]
    
    for i, (true, pred) in enumerate(zip(y_true, y_preds)):
        print(f"Output {i + 1}:")
        make_confusion_matrix(true, pred, classes[i] if classes else None, figsize, text_size, norm, savefig)
        plt.show()

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    # (The function content remains the same as you provided)

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    labels = classes if classes else np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig(f"confusion_matrix_output_{i + 1}.png")
        
        
        
### Attempt at Using the Confusion Matrix Function

def extract_labels_from_dataset(dataset, problem_type):
    """
    Extract labels from a TensorFlow dataset based on the problem type.
    
    Args:
    - dataset (tf.data.Dataset): The TensorFlow dataset to extract labels from.
    - problem_type (str): The type of problem ('Multi-Output', 'Multi-Class', or 'Binary').
    
    Returns:
    - numpy array or dict: If 'Multi-Output', returns a numpy array with shape (num_samples, num_outputs).
                           If 'Multi-Class' or 'Binary', returns a dictionary with label types as keys and 
                           arrays of labels as values.
    """
    
    if problem_type == 'Multi-Output':
        labels_list = [labels for _, labels in dataset]
        return np.array(labels_list).squeeze()

    elif problem_type in ['Multi-Class', 'Binary']:
        labels_dict = {}
        for label_type in ['Focus_Label', 'StigX_Label', 'StigY_Label']:
            label_data = []
            for _, labels in dataset[label_type]:
                label_data.extend(labels.numpy())
            labels_dict[label_type] = np.array(label_data)
        return labels_dict

    else:
        print(f"Unknown problem type: {problem_type}")
        return None

# Usage
problem_type = config['Experiment']['PROBLEM_TYPE']
test_labels = extract_labels_from_dataset(test_dataset, problem_type)

# Choose a specific model (replace 'specific_model_name' with the actual model name you're interested in)
model_name = 'specific_model_name'
model = all_best_models[model_name]

if not model:
    print(f"No model found for {model_name}")
    exit()

# 1. Predict on the test data
predictions = model.predict(test_dataset)


# 2. Get true labels and predictions for each output
# Assuming test_labels is a list where each item is an array of true labels for a given output
true_labels = [test_labels[i] for i in range(len(predictions))]
predicted_labels = [np.argmax(predictions[i], axis=1) for i in range(len(predictions))]

# List of class names for each output, assuming they are the same for all outputs in this example
classes_list = [list(range(3)) for _ in range(len(predictions))]

# 3. Generate confusion matrices
make_confusion_matrix_multi_output(true_labels, predicted_labels, classes_list)


