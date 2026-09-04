# 🐱🐶 Cat vs Dog Image Classification

## 📌 Overview

This project is a Deep Learning image classification system that identifies whether an input image is a **Cat** or a **Dog**.

The project uses **MobileNetV2 Transfer Learning** with TensorFlow and Keras.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## 🧠 Model

A pre-trained **MobileNetV2** model with ImageNet weights is used as the base model.

The base model is frozen and additional classification layers are added for binary Cat/Dog classification.

## 📊 Results

| Metric | Result |
|---|---:|
| Training Images | 20,000 |
| Validation Images | 5,000 |
| Validation Accuracy | **98.32%** |
| Test Images | 10 |
| Test Accuracy | **90%** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Cat | 0.98 | 0.98 | 0.98 |
| Dog | 0.98 | 0.98 | 0.98 |

## 📂 Project Files

- `train_model.py` – trains the MobileNetV2 model
- `test_images.py` – tests individual images
- `evaluation.py` – generates evaluation metrics
- `cat_dog_model.keras` – trained model
- `confusion_matrix.png` – confusion matrix
- `requirements.txt` – required Python libraries

## 🚀 How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
