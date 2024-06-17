import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model('digit_classifier_model.keras')

def load_and_preprocess_image(image_path):
    # Load image
    img = plt.imread(image_path)
    if img.ndim == 3:  # Check if image is RGB
        img = img[:, :, 0]  # Convert to grayscale by taking one channel
    img = np.resize(img, (28, 28))  # Resize the image
    img = img / 255.0  # Normalize the image
    return img

def predict_digit(image_path):
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    img = img.reshape(1, 28, 28)  # Reshape for model input
    # Model prediction
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)  # Return class and confidence

# Example usage
image_path = 'sample_zero.png'  # Path to the image
predicted_class, confidence = predict_digit(image_path)
print(f'Predicted class: {predicted_class}, Confidence: {confidence:.2f}')
