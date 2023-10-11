# model_inspection.py

### Displaying Model Outputs in a Table (Refactored GraphVersion)

from tabulate import tabulate

def predict_with_model(model, sample_input):
    """Predict the output using the given model and input."""
    output = model.predict(sample_input)
    
    # Convert TensorFlow tensor to numpy array if needed
    if hasattr(output, 'numpy'):
        output = output.numpy()
    
    return output

def capture_model_output_details(model_name, model, sample_input):
    """Capture details of model's output for a given input."""
    output = predict_with_model(model, sample_input)
    return {
        "Model Name": model_name,
        "Input Shape": model.input_shape,
        "Output Shape": output.shape,
        "Output Type": type(output).__name__,
        "Sample Output": output[0] if len(output) > 0 else "No Output"
    }

def display_model_outputs_in_table(all_best_models, sample_input):
    """Display details of model's outputs in a tabular format."""
    model_output_details = [capture_model_output_details(model_name, model, sample_input) 
                            for model_name, model in all_best_models.items()]
    
    print(tabulate(model_output_details, headers="keys", tablefmt="grid"))

# Sample Usage
sample_input = test_images
display_model_outputs_in_table(all_best_models, sample_input)


### Inspecting the first few samples from the test dataset (Refactored GraphVersion)

def inspect_test_samples(test_dataset, label_encoders, num_samples_to_inspect=3):
    """
    Inspect a specified number of samples from a test dataset.
    
    Args:
    - test_dataset (tf.data.Dataset): The test dataset to inspect.
    - label_encoders (dict): A dictionary of label encoders.
    - num_samples_to_inspect (int, optional): Number of samples to inspect. Defaults to 3.
    
    Returns:
    - df: Styled DataFrame with inspected samples.
    """
    
    # Load a batch of test images, labels, and offset values
    test_images, test_labels_list, test_offsets = next(iter(test_dataset.take(1)))
    # test_images, test_labels_list = next(iter(test_dataset.take(1)))
    test_labels_array = np.stack([np.array(label) for label in test_labels_list])

    # Prepare data for DataFrame
    data = []
    for i in range(num_samples_to_inspect):
        sample_data = {
            "Sample": i + 1,
            "Image shape": str(test_images[i].shape),
            "Image values (first few)": f"{str(test_images[i].numpy().flatten()[:10])}...",
            "Labels shape": str(test_labels_array[i].shape),
            "Labels values": str(test_labels_array[i]),
        }

        for label, value in zip(label_encoders.keys(), test_labels_array[i]):
            sample_data[f"{label} (Decoded)"] = label_encoders[label].inverse_transform([value])[0]

        data.append(sample_data)

    # Create and display DataFrame
    df = pd.DataFrame(data).set_index("Sample")
    return df.style.hide_index()

# Sample Usage
styled_df = inspect_test_samples(train_dataset, label_encoders)
display(styled_df)

