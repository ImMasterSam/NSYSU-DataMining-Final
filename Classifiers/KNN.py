import numpy as np
import pandas as pd
from Classifiers.classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier(Classifier):

    def __init__(self, k: int, normDistance: int = 2, normalize: bool = True):
        super().__init__('KNN Classifier', normalize)
        self.k = k
        self.normDis = normDistance
        self.clf = KNeighborsClassifier(n_neighbors = k, p = normDistance)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.x_train = x_train.to_numpy()
        self.y_train = y_train.to_numpy().flatten()

        if self.normalize:
            self.x_train = self.scaler.fit_transform(self.x_train)

        self.clf.fit(self.x_train, self.y_train)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.to_numpy()

        if self.normalize:
            x_test = self.scaler.transform(x_test)
        
        y_predict = self.clf.predict(x_test)
        return y_predict