import pandas as pd
import numpy as np
from scipy.stats import mode

class Cluster:

    def __init__(self, name: str):
        self.name = name

    def fit_predict(self, x_train: pd.DataFrame, labels: pd.DataFrame):
        pass
    
    def score(self, x_test: pd.DataFrame, y_classified: pd.DataFrame, y_test: pd.DataFrame, output: bool = False) -> float:
        ''' 輸出準確度 Accuracy '''
        y_predict = self.fit_predict(x_test, y_classified)
        mapped_labels = self.best_map(y_test, y_predict)
        accuracy = np.mean(mapped_labels == y_test)
        if output:
            print(f"{self.name} Score:  {accuracy * 100:.2f} %")
            print("Result:\n", mapped_labels.ravel())
            print()
        return accuracy
    
    def best_map(self, true_labels, cluster_labels):
        ''' 將分群結果映射到真實標籤 '''
        
        # 這裡會有問題
        # 可能會把不同的分群標籤映射到同一個真實標籤上 (或是原本已知的標籤)
        # 需要確保每個分群標籤對應到一個唯一的真實標籤

        label_map = {}
        for c in np.unique(cluster_labels):
            mask = (cluster_labels == c)
            if np.any(mask):
                mapped = mode(true_labels[mask], keepdims=True)[0][0]
                label_map[c] = mapped
        return np.array([label_map[c] for c in cluster_labels])