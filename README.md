# Digit Classifier using TensorFlow and Keras

This project is a binary digit classifier built using TensorFlow and Keras. The classifier is trained to distinguish between the digits '0' and '1' from the MNIST dataset. The project includes data preprocessing, model building, training, evaluation, and prediction on custom images.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Prediction](#prediction)
- [License](#license)

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/digit-classifier.git
    cd digit-classifier
    ```

2. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

    Ensure the `requirements.txt` contains the following:
    ```txt
    tensorflow
    matplotlib
    numpy
    certifi
    ```

## Usage

### Model Training

The script loads the MNIST dataset, filters it for the digits '0' and '1', normalizes the images, builds a neural network model, trains it, and saves the model.

### Model Evaluation

After training, the model's performance on the test dataset is evaluated and printed. The model is saved in the Keras format for future use.

### Prediction

To predict the class of a custom image, use the provided functions. The image must be preprocessed before feeding it into the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
