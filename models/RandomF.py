import numpy as np
import pandas as pd
from models.model import Classifier
from collections import Counter
from sklearn.ensemble import RandomForestClassifier as skRandomForest

class RandomForestClassifier(Classifier):
    def __init__(self, n_estimators: int = 10, max_depth: int = 10, min_samples_split: int = 2, normalize: bool = True):
        super().__init__('Random Forest Classifier', normalize)
        self.clf = skRandomForest(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        if self.normalize:
            self.mean = x_train.mean(axis=0)
            self.std = x_train.std(axis=0)
            x_train = (x_train - self.mean) / self.std

        self.clf.fit(x_train,y_train)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.to_numpy()

        if self.normalize:
            x_test = (x_test - self.mean) / self.std

        y_predict = self.clf.predict(x_test)
        
        return y_predict
