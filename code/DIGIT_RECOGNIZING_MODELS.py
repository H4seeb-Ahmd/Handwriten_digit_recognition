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

def sigmoid(x):
    # The sigmoid function
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of the sigmoid
    return sigmoid(x) * (1 - sigmoid(x))

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
            print(f"Training {self.model_name} Classifier...")
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
        self.model_filename = 'TRAINED/nnl_model.bin'
        self.model_name = "Neural Network (Library)"  

        X_train = X_train_raw.reshape(X_train_raw.shape[0], 28 * 28)
        X_test = X_test_raw.reshape(X_test_raw.shape[0], 28 * 28)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        if os.path.exists(self.model_filename):
            with open(self.model_filename, 'rb') as file:
                self.nn_clf = pickle.load(file)
        else:
            print(f"Training {self.model_name} Classifier...")
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

class NeuralNetwork:
    def __init__(self, sizes=[784, 128, 64, 10], epochs = 20, learning_rate=3.0, batch_size=10):
        self.model_filename = 'TRAINED/' + 'nn_model' + str(hash((tuple(sizes), epochs, learning_rate, batch_size))) + '.bin'
        self.model_name = "Neural Network"

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.X_train = [x.flatten().reshape(-1, 1) / 255.0 for x in X_train_raw]
        self.X_test = [x.flatten().reshape(-1, 1) / 255.0 for x in X_test_raw]

        def vectorized_result(j):
            e = np.zeros((10, 1))
            e[j] = 1.0
            return e
        
        y_train_vec = [vectorized_result(y) for y in y_train]

        self.training_data = list(zip(self.X_train, y_train_vec))
        self.test_data = list(zip(self.X_test, y_test))

        if os.path.exists(self.model_filename):
            with open(self.model_filename, 'rb') as file:
                saved_data = pickle.load(file)
                self.weights = saved_data['weights']
                self.biases = saved_data['biases']
        else:
            print(f"Training {self.model_name}...")
            # Initialize random weights and biases
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            
            # Train the network
            self._train()
            
            # Evaluate Accuracy
            accuracy = self._evaluate(self.test_data)
            print(f"{self.model_name} Accuracy: {accuracy / len(self.test_data) * 100:.2f}%")
            
            # Save the trained weights and biases
            with open(self.model_filename, 'wb') as file:
                pickle.dump({'weights': self.weights, 'biases': self.biases}, file)
        

        print(f"{self.model_name} model loaded successfully!")

    def _train(self):
        """Trains the network using mini-batch gradient descent."""
        n = len(self.training_data)
        for j in range(self.epochs):
            # Shuffle data to prevent patterns from affecting training
            np.random.shuffle(self.training_data)
            
            # Create mini-batches for faster, more stable training
            mini_batches = [self.training_data[k:k+self.batch_size] 
                            for k in range(0, n, self.batch_size)]
            
            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch)

    def _update_mini_batch(self, mini_batch):
        """Update weights and biases by applying gradient descent to a single mini-batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        # Adjust weights and biases based on the accumulated gradients
        self.weights = [w - (self.learning_rate / len(mini_batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.learning_rate / len(mini_batch)) * nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def _backprop(self, x, y):
        """The core math from the video: returns the gradient for the cost function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward
        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # Backward pass
        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        return (nabla_b, nabla_w)

    def _feedforward(self, a):
        """Pumps the input through the network."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def _evaluate(self, test_data):
        """Checks how many test images the network guesses correctly."""
        test_results = [(np.argmax(self._feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def model_prediction(self, entry):
        """Format matching your KNN prediction method."""
        # 1. Normalize and reshape the incoming entry just like training data
        entry_reshaped = entry.flatten().reshape(-1, 1) / 255.0
        
        # 2. Get the 10 probability outputs
        output = self._feedforward(entry_reshaped)
        
        # 3. Return the index (0-9) of the highest probability
        return (np.argmax(output),)

