import numpy as np
import pandas as pd
from models.model import Classifier

class NeuralNetClassifier(Classifier):

    def __init__(self, n_hidden: int = 10, learning_rate: float = 0.001, n_iters: int = 1000, normalize: bool = True):
        super().__init__('Neural Network Classifier', normalize)
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.n_iters = n_iters
        self.params = {}

    def _sigmoid(self, z):
        ''' Sigmoid 激活函數 '''
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        ''' Sigmoid 激活函數 -> 導函數 '''
        s = self._sigmoid(z)
        return s * (1 - s)

    def _initialize_weights(self, n_features):
        ''' 初始化參數 '''
        self.params['W1'] = np.random.randn(n_features, self.n_hidden)
        self.params['b1'] = np.zeros((1, self.n_hidden))
        self.params['W2'] = np.random.randn(self.n_hidden, 1)
        self.params['b2'] = np.zeros((1, 1))

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        x_train = x_train.astype(float).to_numpy()
        y_train = y_train.astype(int).to_numpy().reshape(-1, 1)  # 確保 y_train 是列向量

        if self.normalize:
            self.mean = x_train.mean()
            self.std = x_train.std()
            x_train = (x_train - self.mean) / self.std

        n_samples, n_features = x_train.shape
        self._initialize_weights(n_features)

        for _ in range(self.n_iters):
            # Forward pass
            Z1 = np.dot(x_train, self.params['W1']) + self.params['b1']
            A1 = self._sigmoid(Z1)
            Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
            A2 = self._sigmoid(Z2)

            # 計算 loss (交叉熵)
            loss = -np.mean(y_train * np.log(A2 + 1e-8) + (1 - y_train) * np.log(1 - A2 + 1e-8))

            # Backward pass
            dZ2 = A2 - y_train
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, self.params['W2'].T)
            dZ1 = dA1 * self._sigmoid_derivative(Z1)
            dW1 = np.dot(x_train.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # Gradient descent update
            self.params['W1'] -= self.lr * dW1
            self.params['b1'] -= self.lr * db1
            self.params['W2'] -= self.lr * dW2
            self.params['b2'] -= self.lr * db2

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.to_numpy().astype(float)
        if self.normalize:
            x_test = (x_test - self.mean) / self.std

        Z1 = np.dot(x_test, self.params['W1']) + self.params['b1']
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self._sigmoid(Z2)
        predictions = (A2 > 0.5).astype(int).flatten()
        return predictions
