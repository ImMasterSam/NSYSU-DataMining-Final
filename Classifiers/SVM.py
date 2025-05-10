import numpy as np
import pandas as pd
from Classifiers.classifier import Classifier
from sklearn.svm import SVC

class SVMClassifier(Classifier):
    def __init__(self, kernel: str = "rbf", normalize: bool = True):
        super().__init__(f'Kernel SVM ({kernel}) Classifier', normalize)
        self.clf = SVC(kernel = kernel)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.astype(float).to_numpy()
        y_train = y_train.to_numpy().ravel()

        if self.normalize:
            x_train = self.scaler.fit_transform(x_train)

        self.clf.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.astype(float).to_numpy()

        if self.normalize:
            x_test = self.scaler.transform(x_test)

        y_predict = self.clf.predict(x_test)

        return y_predict