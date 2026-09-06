import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐱🐶 Cat & Dog Image Classification")
st.write("Upload an image and the model will predict whether it is a Cat or Dog.")

model = tf.keras.models.load_model("cat_dog_model.keras")

uploaded_file = st.file_uploader(
    "Choose a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("🐶 Prediction: Dog")
    else:
        st.success("🐱 Prediction: Cat")