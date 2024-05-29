import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Загрузка модели
model = load_model('digit_classifier_model.keras')

def load_and_preprocess_image(image_path):
    # Загрузка изображения
    img = plt.imread(image_path)
    if img.ndim == 3:  # Проверка, является ли изображение цветным
        img = img[:, :, 0]  # Преобразование в градации серого, берем только один канал
    img = np.resize(img, (28, 28))  # Изменение размера изображения
    img = img / 255.0  # Нормализация
    return img

def predict_digit(image_path):
    # Загрузка и предобработка изображения
    img = load_and_preprocess_image(image_path)
    img = img.reshape(1, 28, 28)  # Изменение формы для подачи в модель
    # Предсказание модели
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)  # Возвращаем класс и вероятность

# Пример использования
image_path = 'sample_zero.png'  # Путь к изображению
predicted_class, confidence = predict_digit(image_path)
print(f'Predicted class: {predicted_class}, Confidence: {confidence:.2f}')
