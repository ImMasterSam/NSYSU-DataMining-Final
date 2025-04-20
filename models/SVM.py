import numpy as np
import pandas as pd
from models.model import Classifier

class SVMClassifier(Classifier):

    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000, normalize: bool = True):
        super().__init__('Linear SVM Classifier', normalize)
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None #公式的權重
        self.b = None #偏差
        pass

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        x_train = x_train.astype(float).to_numpy()
        y_train = y_train.astype(int).to_numpy()

        if self.normalize:
            self.mean = x_train.mean()
            self.std = x_train.std()
            x_train = (x_train - self.mean) / self.std

        n_samples, n_features = x_train.shape   # sameple 為行 feature為列
        self.w = np.random.randn(n_features)    # 因為svm最後要表達為f(x) = w_1*x1 + w_2*x2 + ...... + w_8*x8 + b 所以初始化向量的特徵
        self.b = 0

        y_t = np.where(y_train <= 0 , -1 ,1)    # 這個公式在將outcome傳換成-1 1因為svm是用-1 1做分類的 y_train <= 0 就是-1 else 1

        for _ in range(self.n_iters):           # 跑迴圈 不斷學習
            for idx, x_i in enumerate(x_train):
                margin = y_t[idx] * (np.dot(x_i, self.w) + self.b)
                if margin < 1:
                    self.w += self.lr * y_t[idx] * x_i
                    self.b += self.lr * y_t[idx]

    def predict(self , x_test: pd.DataFrame):
        x_test = x_test.astype(float).to_numpy()

        if self.normalize:
            x_test = (x_test - self.mean) / self.std

        #分類 (x.w + b) 內積 + bias
        classification = np.dot(x_test, self.w)+self.b
        return np.where(classification <= 0, 0, 1)
        