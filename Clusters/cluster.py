import pandas as pd
import numpy as np
from itertools import permutations

class Cluster:

    def __init__(self, name: str):
        self.name = name

    def fit_predict(self, x_train: pd.DataFrame, labels: pd.DataFrame):
        pass
    
    def score(self, x_test: pd.DataFrame, y_classified: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, output: bool = False) -> float:
        ''' 輸出準確度 Accuracy '''
        y_test = np.array(y_test).ravel()
        print(y_classified)
        y_predict = np.array(self.fit_predict(x_test, y_classified)).ravel()
        print(y_predict)
        mapped_labels = self.best_map(y_test, y_predict, y_train)
        accuracy = np.mean(mapped_labels == y_test)
        if output:
            print(f"{self.name} Score:  {accuracy * 100:.2f} %")
            print("Result:\n", mapped_labels.ravel())
            print()
        return accuracy
    
    def best_map(self, true_labels, cluster_labels, train_labels):
        '''
        用排列組合窮舉所有可能對應，找出最大正確配對數的分群標籤對應。
        分群標籤大於100的唯一一對一對應到未用過的編號。
        '''
        # 將輸入轉換為一維數組
        true_labels = np.array(true_labels).ravel()
        cluster_labels = np.array(cluster_labels).ravel()
        train_labels = np.array(train_labels).ravel()

        known_labels = sorted(set(train_labels))
        cluster_ids = [c for c in np.unique(cluster_labels) if c >= 100]

        unknown_true_labels = set(true_labels) - set(known_labels)

        max_known_label = max(known_labels)
        max_unknown_true_label = max(unknown_true_labels)

        n_true = max_unknown_true_label - max_known_label
        n_cluster = len(cluster_ids)

        # 建立混淆矩陣
        count_matrix = np.zeros((n_cluster, max(n_true, n_cluster)), dtype=int)
        for i, c in enumerate(cluster_ids):
            for j, t in enumerate(range(max_known_label + 1, max_unknown_true_label + 1)):
                count_matrix[i, j] = np.sum((cluster_labels == c) & (true_labels == t))

        # 用排列組合窮舉所有對應
        best_map = None
        best_score = -1
        for perm in permutations(range(max(n_true, n_cluster)), n_cluster):
            score = 0
            for i, j in enumerate(perm):
                score += count_matrix[i, j]
            if score > best_score:
                best_score = score
                best_map = perm

        # 根據最佳配對，將 cluster_id 對應到 known_label
        label_map = {}
        for i, j in enumerate(best_map):
            label_map[cluster_ids[i]] = len(known_labels) + (j + 1)

        # 其他分群標籤（<100）直接對應原標籤
        for c in np.unique(cluster_labels):
            if c < 100:
                label_map[c] = c
        return np.array([label_map[c] for c in cluster_labels])