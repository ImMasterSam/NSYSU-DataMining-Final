import numpy as np
import pandas as pd
from models.model import Classifier
from sklearn import svm

class SVMClassifierWithKernel(Classifier):
    def __init__(self, kernel: str = "rbf", C: float = 3, gamma: float = 0.2, n_iters: int = 1000, normalize: bool = True):
        super().__init__('Kernel SVM Classifier', normalize)
        self.clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.astype(float).to_numpy()
        y_train = np.where(y_train <= 0, -1, 1)

        if self.normalize:
            self.mean = x_train.mean(axis=0)
            self.std = x_train.std(axis=0)
            x_train = (x_train - self.mean) / self.std

        self.clf.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.astype(float).to_numpy()

        if self.normalize:
            x_test = (x_test - self.mean) / self.std

        y_predict = self.clf.predict(x_test)
        y_predict = np.where(y_predict < 0, 0, 1)

        return y_predict