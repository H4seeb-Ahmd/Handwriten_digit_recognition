import idx2numpy
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

# 1. Load the raw files
X_train_raw = idx2numpy.convert_from_file('DATASET/train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('DATASET/train-labels.idx1-ubyte')
X_test_raw = idx2numpy.convert_from_file('DATASET/t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('DATASET/t10k-labels.idx1-ubyte')

# 2. Flatten the images from (60000, 28, 28) to (60000, 784)
X_train = X_train_raw.reshape(X_train_raw.shape[0], 28 * 28)
X_test = X_test_raw.reshape(X_test_raw.shape[0], 28 * 28)

print("Training MEAN Classifier...")
mean_clf = NearestCentroid()
mean_clf.fit(X_train, y_train)

mean_preds = mean_clf.predict(X_test)
mean_acc = accuracy_score(y_test, mean_preds)
print(f"MEAN Classifier Accuracy: {mean_acc * 100:.2f}%")