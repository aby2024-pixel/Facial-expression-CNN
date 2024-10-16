# train.py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import model as mdl

# Load and preprocess dataset (Example: using FER2013 dataset)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fer2013.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize data and reshape
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 48, 48, 1)
X_val = X_val.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# Convert labels to categorical
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build and compile model
input_shape = (48, 48, 1)
model = mdl.build_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_val, y_val))

# Save model
model.save('facial_expression_model.h5')
