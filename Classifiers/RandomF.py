import numpy as np
import pandas as pd
from Classifiers.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier as skRandomForest
from sklearn.base import BaseEstimator

class RandomForestClassifier(Classifier, BaseEstimator):
    def __init__(self,
                 n_estimators: int = 10,
                 max_depth: int = 10,
                 min_samples_split: int = 2,
                 normalize: bool = True,
                 proba: bool = False,
                 threshold: float = 0.6):
        super().__init__('Random Forest Classifier', normalize, proba, threshold)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.clf = skRandomForest(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, n_jobs = -1)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.to_numpy()
        y_train = np.array(y_train).ravel()
        
        if self.normalize:
            x_train = self.scaler.fit_transform(x_train)
        self.clf.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.to_numpy()

        if self.normalize:
            x_test = self.scaler.transform(x_test)

        if self.proba:
            y_proba = self.clf.predict_proba(x_test)

            # 找出最大機率的類別及其機率
            max_proba = np.max(y_proba, axis=1)
            max_class_idx = np.argmax(y_proba, axis=1)
            # 只保留機率大於 threshold 的標籤，否則設為 None 或其他標記
            y_predict = np.where(max_proba >= self.threshold,
                                 self.clf.classes_[max_class_idx], -1)
        else:
            y_predict = self.clf.predict(x_test)
        
        return y_predict

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'normalize': self.normalize,
            'proba': self.proba,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.clf = skRandomForest(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        return self
