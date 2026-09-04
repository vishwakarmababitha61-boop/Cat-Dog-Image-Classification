import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Dataset
dataset_path = "dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="binary",
    image_size=(160, 160),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="binary",
    image_size=(160, 160),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Data performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# MobileNetV2
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Model
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Evaluate
loss, accuracy = model.evaluate(val_ds)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print("Validation Accuracy:", accuracy)
print("Validation Loss:", loss)

# Save
model.save("cat_dog_model.keras")

print("\nTraining completed!")
print("Model saved as cat_dog_model.keras")