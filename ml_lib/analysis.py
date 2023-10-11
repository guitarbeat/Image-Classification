# analysis.py

# Analyzing Most Wrong Predictions:
# Observing predictions with high confidence but incorrect can provide insights into model performance.
# Steps:
# 1. Get test image filepaths.
# 2. Create a DataFrame for image details and predictions.
# 3. Identify correct predictions.
# 4. Extract top 100 most confidently incorrect predictions.
# 5. Visualize some of the most wrong examples.

# 1. Get the filenames of all test data
filepaths = [filepath.numpy() for filepath in test_data.list_files("101_food_classes_10_percent/test/*/*.jpg", shuffle=False)]

# 2. Create a dataframe for prediction analysis
import pandas as pd
pred_df = pd.DataFrame({
    "img_path": filepaths,
    "y_true": y_labels,
    "y_pred": pred_classes,
    "pred_conf": pred_probs.max(axis=1),
    "y_true_classname": [class_names[i] for i in y_labels],
    "y_pred_classname": [class_names[i] for i in pred_classes]
})

# 3. Identify correct predictions
pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]

# 4. Extract top 100 confidently incorrect predictions
top_100_wrong = pred_df[pred_df["pred_correct"] == False].sort_values("pred_conf", ascending=False).head(100)

# 5. Visualize some of the most wrong examples
images_to_view = 9
plt.figure(figsize=(15, 10))
for i, row in enumerate(top_100_wrong.head(images_to_view).itertuples()): 
    plt.subplot(3, 3, i+1)
    img = load_and_prep_image(row.img_path, scale=True)
    plt.imshow(img)
    plt.title(f"actual: {row.y_true_classname}, pred: {row.y_pred_classname} \nprob: {row.pred_conf:.2f}")
    plt.axis(False)


