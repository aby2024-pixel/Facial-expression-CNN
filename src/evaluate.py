# evaluate.py
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load test data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fer2013.load_data()
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 48, 48, 1)

# Load model
model = tf.keras.models.load_model('facial_expression_model.h5')

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Generate confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test

conf_matrix = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.show()
