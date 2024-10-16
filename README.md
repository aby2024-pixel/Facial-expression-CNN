# Facial-expression-CNN
# Facial Expression Recognition using Convolutional Neural Networks (CNNs)

## Project Overview
This project aims to recognize facial expressions in images using Convolutional Neural Networks (CNNs). The project is implemented in Python using TensorFlow/Keras, and evaluates the performance using various metrics such as accuracy and confusion matrices.

## Project Structure
- `model.py`: Defines the CNN model architecture.
- `train.py`: Loads the dataset, preprocesses it, trains the CNN model, and saves it.
- `evaluate.py`: Loads the trained model and evaluates it on the test data, including generating a confusion matrix.
- `requirements.txt`: Lists the dependencies for the project.

## Installation
To install the dependencies, run:
```
pip install -r requirements.txt
```

## Usage
1. Train the model:
```
python train.py
```
2. Evaluate the model:
```
python evaluate.py
```

## Dataset
The FER2013 dataset is used for training and testing the model. The dataset contains 48x48 pixel grayscale images of facial expressions.

## Results
The model is evaluated based on accuracy, and a confusion matrix is generated to visualize the performance across different facial expressions.

## License
This project is licensed under the MIT License
