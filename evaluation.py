import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.keras")

# Load validation dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    labels="inferred",
    label_mode="binary",
    image_size=(160, 160),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

y_true = []
y_pred = []

# Make predictions
for images, labels in dataset:

    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten())
    y_pred.extend(
        (predictions.flatten() >= 0.5).astype(int)
    )

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

# Classification Report
print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["cats", "dogs"]
    )
)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["cats", "dogs"]
)

disp.plot()

plt.title("Cat vs Dog Confusion Matrix")
plt.tight_layout()

# Save image
plt.savefig("confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved as confusion_matrix.png")