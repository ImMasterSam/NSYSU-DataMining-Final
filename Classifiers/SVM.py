import numpy as np
import pandas as pd
from Classifiers.classifier import Classifier
from sklearn.svm import SVC

class SVMClassifier(Classifier):
    def __init__(self, kernel: str = "rbf", normalize: bool = True, proba: bool = False, threshold: float = 0.6):
        super().__init__(f'Kernel SVM ({kernel}) Classifier', normalize, proba, threshold)
        self.clf = SVC(kernel = kernel, probability = proba, class_weight = 'balanced')

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