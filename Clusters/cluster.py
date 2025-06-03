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
        y_test = np.array(y_test).ravel()
        y_predict = np.array(self.fit_predict(x_test, y_classified)).ravel()
        mapped_labels = self.best_map(y_test, y_predict)
        accuracy = np.mean(mapped_labels == y_test)
        if output:
            print(f"{self.name} Score:  {accuracy * 100:.2f} %")
            print("Result:\n", mapped_labels.ravel())
            print()
        return accuracy
    
    def best_map(self, true_labels, cluster_labels):
        ''' 將分群結果映射到真實標籤，分群標籤大於100的唯一一對一對應到未用過的編號 '''
        true_labels = np.array(true_labels).ravel()
        cluster_labels = np.array(cluster_labels).ravel()
        # 已知真實標籤
        known_labels = sorted(set(true_labels))
        next_label = max(known_labels) + 1 if known_labels else 1
        # 只處理大於100的分群標籤
        cluster_ids = [c for c in np.unique(cluster_labels) if c >= 100]
        # 建立混淆矩陣
        count_matrix = np.zeros((len(cluster_ids), len(known_labels)), dtype=int)
        for i, c in enumerate(cluster_ids):
            for j, t in enumerate(known_labels):
                count_matrix[i, j] = np.sum((cluster_labels == c) & (true_labels == t))
        # 貪婪法：每次找最大值，並將該行列設為-1，確保一對一
        label_map = {}
        used_true = set()
        used_cluster = set()
        for _ in range(min(len(cluster_ids), len(known_labels))):
            idx = np.unravel_index(np.argmax(count_matrix), count_matrix.shape)
            ci, tj = idx
            if count_matrix[ci, tj] == 0:
                break
            label_map[cluster_ids[ci]] = known_labels[tj]
            used_true.add(known_labels[tj])
            used_cluster.add(cluster_ids[ci])
            count_matrix[ci, :] = -1
            count_matrix[:, tj] = -1
        # 剩下未配對的分群標籤，從 next_label 開始往上編號
        for c in cluster_ids:
            if c not in label_map:
                label_map[c] = next_label
                next_label += 1
        # 其他分群標籤（<=100）直接對應原標籤
        for c in np.unique(cluster_labels):
            if c < 100:
                label_map[c] = c
        return np.array([label_map[c] for c in cluster_labels])