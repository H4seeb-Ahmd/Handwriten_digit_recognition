import idx2numpy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import pickle


variables_filename = 'dataset.npz'

if os.path.exists(variables_filename):
    data = np.load(variables_filename)
    X_train_raw = data['X_train']
    y_train = data['y_train']
    X_test_raw = data['X_test']
    y_test = data['y_test']
else:
    X_train_raw = idx2numpy.convert_from_file('DATASET/train-images.idx3-ubyte')
    y_train = idx2numpy.convert_from_file('DATASET/train-labels.idx1-ubyte')
    X_test_raw = idx2numpy.convert_from_file('DATASET/t10k-images.idx3-ubyte')
    y_test = idx2numpy.convert_from_file('DATASET/t10k-labels.idx1-ubyte')
    
    np.savez(variables_filename, 
             X_train=X_train_raw, 
             y_train=y_train, 
             X_test=X_test_raw, 
             y_test=y_test)

class KNN:
    def __init__(self):
        self.model_filename = 'TRAINED/knn_model.bin'
        self.model_name = "KNN"
        
        X_train = X_train_raw.reshape(X_train_raw.shape[0], 28 * 28)
        X_test = X_test_raw.reshape(X_test_raw.shape[0], 28 * 28)

        if os.path.exists(self.model_filename):
            with open(self.model_filename, 'rb') as file:
                self.knn_clf = pickle.load(file)
        else:   
            print("Training KNN Classifier...")
            self.knn_clf = KNeighborsClassifier()
            self.knn_clf.fit(X_train, y_train)
            knn_preds = self.knn_clf.predict(X_test)
            knn_acc = accuracy_score(y_test, knn_preds)
            print(f"KNN Classifier Accuracy: {knn_acc * 100:.2f}%")
            with open(self.model_filename, 'wb') as file:
                pickle.dump(self.knn_clf, file)
        print(f"{self.model_name} model loaded successfully!")
    def model_prediction(self, entry):
            return self.knn_clf.predict(entry)

class NeuralNetworkLib:
    def __init__(self):
        self.model_filename = 'TRAINED/nn_model.bin'
        self.model_name = "Neural Network (Library)"  

        X_train = X_train_raw.reshape(X_train_raw.shape[0], 28 * 28)
        X_test = X_test_raw.reshape(X_test_raw.shape[0], 28 * 28)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        if os.path.exists(self.model_filename):
            with open(self.model_filename, 'rb') as file:
                self.nn_clf = pickle.load(file)
        else:
            print("Training KNN Classifier...")
            self.nn_clf = MLPClassifier(
                hidden_layer_sizes = (128,),
                activation = 'relu',
                solver = 'adam',
                max_iter = 20
            )

            self.nn_clf.fit(X_train, y_train)

            mlp_preds = self.nn_clf.predict(X_test)
            mlp_acc = accuracy_score(y_test, mlp_preds)
            print(f"Neural Network Accuracy: {mlp_acc * 100:.2f}%")
            
            with open(self.model_filename, 'wb') as file:
                pickle.dump(self.nn_clf, file)
                
        print(f"{self.model_name} model loaded successfully!")

    def model_prediction(self, entry):
            return self.nn_clf.predict(entry)
