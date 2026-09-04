import tensorflow as tf
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.keras")

# Test images folder
folder = "test_images"

correct = 0
total = 0

print("\n==============================")
print("CAT vs DOG TEST RESULTS")
print("==============================\n")

for file in os.listdir(folder):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(folder, file)

        # Load image
        image = tf.keras.utils.load_img(
            path,
            target_size=(160, 160)
        )

        # Convert image to array
        image_array = tf.keras.utils.img_to_array(image)

        # IMPORTANT:
        # Do NOT divide by 255 because the model
        # already performs Rescaling internally.
        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        prediction = model.predict(image_array, verbose=0)[0][0]

        # Convert prediction to class
        if prediction >= 0.5:
            predicted = "Dog"
        else:
            predicted = "Cat"

        # Get actual label from filename
        if file.lower().startswith("dog"):
            actual = "Dog"
        else:
            actual = "Cat"

        # Check correctness
        if actual == predicted:
            correct += 1

        total += 1

        print(
            f"{file} | Actual: {actual} | Predicted: {predicted}"
        )

# Accuracy on 10 test images
accuracy = (correct / total) * 100

print("\n==============================")
print("TEST SUMMARY")
print("==============================")
print(f"Correct Predictions: {correct}/{total}")
print(f"Test Accuracy: {accuracy:.2f}%")
print("\nTesting completed!")