import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
import ssl
import certifi

# SSL context setup for data downloading
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Load and filter MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
filter_train = (train_labels == 0) | (train_labels == 1)
filter_test = (test_labels == 0) | (test_labels == 1)
train_images, train_labels = train_images[filter_train], train_labels[filter_train]
test_images, test_labels = test_images[filter_test], test_labels[filter_test]

# Data normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Точность на тестовой выборке:', test_acc)

# Save the model in Keras format

model.save('digit_classifier_model.keras')

# Function for predicting based on a loaded image
def predict_digit(image):
    prediction = model.predict(image.reshape(1, 28, 28))
    return np.argmax(prediction)

