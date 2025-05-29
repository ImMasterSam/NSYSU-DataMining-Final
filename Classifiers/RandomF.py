import numpy as np
import pandas as pd
from Classifiers.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier as skRandomForest

class RandomForestClassifier(Classifier):
    def __init__(self, n_estimators: int = 10, max_depth: int = 10, min_samples_split: int = 2, normalize: bool = True, proba: bool = False, threshold: float = 0.6):
        super().__init__('Random Forest Classifier', normalize, proba, threshold)
        self.clf = skRandomForest(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split)

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy().ravel()

        if self.normalize:
            x_train = self.scaler.fit_transform(x_train)

        self.clf.fit(x_train,y_train)

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
