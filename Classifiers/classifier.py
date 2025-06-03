import numpy as np
import pandas as pd
import prettytable as pt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class Classifier:

    def __init__(self, name: str, normalize: bool = True, proba: bool = False, threshold: float = 0.6):
        self.name = name
        self.normalize = normalize
        self.scaler = StandardScaler()

        self.proba = proba
        self.threshold = threshold

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    def predict(self, x_test: pd.DataFrame):
        pass

    def score(self, x_test: pd.DataFrame, y_test : pd.Series, output: bool = False) -> float:
        ''' 輸出準確度 Accuracy '''
        y_predict = self.predict(x_test)
        known_mask = (y_predict != -1)
        accuracy = np.mean(y_predict[known_mask] == y_test[known_mask].to_numpy().astype(int))
        if output:
            print(f"{self.name} Score:  {accuracy * 100:.2f} %", end=' ')
            print("(with known labels)" if self.proba else "")
            # print(y_predict)
        return accuracy

    def analysis(self, x_test: pd.DataFrame, y_test: pd.DataFrame, y_train: pd.DataFrame, output: bool = False) -> dict:
        ''' 輸出各項測量值 '''
        
        known_classes = {*map(int, y_train.iloc[:, 0].unique())}
        y_labels = y_test.iloc[:, 0].apply(lambda x: x if x in known_classes else -1)
        
        y_predict = self.predict(x_test)
        
        report_str = classification_report(y_labels, y_predict, output_dict=False, zero_division=0)
        report_dict = classification_report(y_labels, y_predict, output_dict=True, zero_division=0)

        if output:
            print(f"\t--- {self.name} Analysis ---")
            print(report_str)
        else:
            print(f"{self.name} Accuracy = {report_dict['accuracy'] * 100:.2f} %")

        return report_dict
        

       