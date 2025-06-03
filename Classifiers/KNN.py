import numpy as np
import pandas as pd
from Classifiers.classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

class KNNClassifier(Classifier, BaseEstimator):

    def __init__(self,
                 normalize: bool = True,
                 proba: bool = True,
                 threshold: float = 0.6,
                 k: int = 21,
                 normDistance: int = 2,
                 weights: str = 'uniform'):
        super().__init__('KNN Classifier', normalize, proba, threshold)
        self.k = k
        self.normDis = normDistance
        self.clf = KNeighborsClassifier(n_neighbors = k, p = normDistance, weights = weights)

    def fit(self, x_train: pd.DataFrame, y_train):
        self.x_train = x_train.to_numpy()
        self.y_train = np.array(y_train).ravel()
        
        if self.normalize:
            self.x_train = self.scaler.fit_transform(self.x_train)
        self.clf.fit(self.x_train, self.y_train)

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
            'normalize': self.normalize,
            'proba': self.proba,
            'threshold': self.threshold,
            'k': self.k,
            'normDistance': self.normDis,
            'weights': self.clf.get_params().get('weights', 'uniform')
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # 重新建立 sklearn 的 KNeighborsClassifier
        self.clf = KNeighborsClassifier(n_neighbors=self.k, p=self.normDis)
        return self