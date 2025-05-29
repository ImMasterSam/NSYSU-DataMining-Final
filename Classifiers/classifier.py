import numpy as np
import pandas as pd
import prettytable as pt
import time
from sklearn.preprocessing import StandardScaler

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
        accuracy = np.mean(y_predict == y_test.to_numpy().astype(int))
        if output:
            print(f"{self.name} Score:  {accuracy * 100:.2f} %")
            # print(y_predict)
        return accuracy

    def analysis(self, x_test: pd.DataFrame, y_test : pd.Series, output: bool = False) -> dict:
        ''' 輸出各項測量值 '''

        start_time = time.time()
        # 執行一些程式碼
        end_time = time.time()

        measures = {}

        table = pt.PrettyTable()
        field_names = ['Classifier', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F-Score', 'Accuracy']
        table.field_names = field_names

        start_time = time.time()
        y_predict = self.predict(x_test)
        end_time = time.time()
        predict_time = end_time - start_time

        p = np.sum(y_test == 1)
        n = np.sum(y_test == 0)
        all = p + n
        tp = np.sum((y_test == 1) & (y_predict == 1))
        fp = np.sum((y_test == 0) & (y_predict == 1))
        tn = np.sum((y_test == 0) & (y_predict == 0))
        fn = np.sum((y_test == 1) & (y_predict == 0))

        measures['Predict time'] = predict_time
        measures['Sensitivity'] = tp / p
        measures['Specificity'] = tn / n
        measures['Precision'] = tp / (tp + fp)
        measures['Recall'] = tn / (tn + fn)
        measures['F-Score'] = (2 * measures['Precision'] * measures['Recall']) / (measures['Precision'] + measures['Recall'])
        measures['Accuracy'] = (tp + tn) / all
        
        table.add_row([self.name] + [*(str(round(measures[s] * 100, 2)) + " %" for s in field_names[1:])])

        if output:
            print(table)

        return measures
        

       