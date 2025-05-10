import numpy as np
import pandas as pd
from Classifiers.classifier import Classifier

class NeuralNetClassifier(Classifier):

    def __init__(self, n_hidden: int = 10, learning_rate: float = 0.001, n_iters: int = 1000, normalize: bool = True):
        super().__init__('Neural Network Classifier', normalize)
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.n_iters = n_iters
        self.params = {}
        self.n_classes = None  # 儲存類別數量
        self.class_mapping = {}  # 用於映射類別標籤

    def _sigmoid(self, z):
        ''' Sigmoid 激活函數 '''
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        ''' Sigmoid 激活函數 -> 導函數 '''
        s = self._sigmoid(z)
        return s * (1 - s)
        
    def _softmax(self, z):
        ''' Softmax 激活函數 '''
        # 為了數值穩定性，減去每一行的最大值
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _initialize_weights(self, n_features):
        ''' 初始化參數 '''
        self.params['W1'] = np.random.randn(n_features, self.n_hidden)
        self.params['b1'] = np.zeros((1, self.n_hidden))
        self.params['W2'] = np.random.randn(self.n_hidden, self.n_classes)
        self.params['b2'] = np.zeros((1, self.n_classes))

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        x_train = x_train.astype(float).to_numpy()
        y_train = y_train.astype(int).to_numpy()  # 確保 y_train 是列向量

        if self.normalize:
            x_train = self.scaler.fit_transform(x_train)

        n_samples, n_features = x_train.shape
        # 取得唯一類別值並確保從0開始編碼
        unique_classes = np.unique(y_train)
        self.n_classes = len(unique_classes)  # 設置類別數量
        
        # 重新映射標籤，確保從0開始
        self.class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        mapped_y_train = np.array([self.class_mapping[y[0]] for y in y_train])
        
        # 使用重新映射後的標籤進行 one-hot 編碼
        y_train_one_hot = np.eye(self.n_classes)[mapped_y_train]  # One-hot 編碼

        self._initialize_weights(n_features)

        for _ in range(self.n_iters):
            # Forward pass
            Z1 = np.dot(x_train, self.params['W1']) + self.params['b1']
            A1 = self._sigmoid(Z1)
            Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
            A2 = self._softmax(Z2)

            # 計算 loss (交叉熵)
            loss = -np.mean(np.sum(y_train_one_hot * np.log(A2 + 1e-8), axis=1))

            # Backward pass
            dZ2 = A2 - y_train_one_hot
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
            x_test = self.scaler.transform(x_test)

        Z1 = np.dot(x_test, self.params['W1']) + self.params['b1']
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self._softmax(Z2)
        predictions = np.argmax(A2, axis=1)
        
        # 將數字預測映射回原始類別標籤
        if hasattr(self, 'class_mapping'):
            # 反轉映射字典
            inv_mapping = {v: k for k, v in self.class_mapping.items()}
            # 將預測映射回原始類別標籤
            predictions = np.array([inv_mapping[pred] for pred in predictions])
            
        return predictions
