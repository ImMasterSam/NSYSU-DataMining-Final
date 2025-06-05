import numpy as np
import pandas as pd
from Clusters.cluster import Cluster

class DBScanCluster(Cluster):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__(name='DBScan Cluster')
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit_predict(self, x_dataset, labels):
        labels = np.array(labels).ravel()
        unknown_mask = (labels == -1)
        x_unknown = x_dataset[unknown_mask]
        x_unknown = np.array(x_unknown)

        # 資料標準化
        if len(x_unknown) > 0:
            x_unknown = (x_unknown - x_unknown.mean(axis=0)) / (x_unknown.std(axis=0) + 1e-8)

        # 若沒有未知資料，直接回傳原標籤
        if len(x_unknown) == 0:
            return labels

        n_points = len(x_unknown)
        cluster_labels = np.full(n_points, -1, dtype=int)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0

        def region_query(point_idx):
            ''' 找到在 eps 範圍內的鄰居點 '''
            dists = np.linalg.norm(x_unknown - x_unknown[point_idx], axis=1)
            return np.where(dists <= self.eps)[0]

        def expand_cluster(point_idx, neighbors, cluster_id):
            ''' 擴展分群 '''
            cluster_labels[point_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                n_idx = neighbors[i]
                if not visited[n_idx]:
                    visited[n_idx] = True
                    n_neighbors = region_query(n_idx)
                    if len(n_neighbors) >= self.min_samples:
                        neighbors = np.concatenate((neighbors, n_neighbors))
                if cluster_labels[n_idx] == -1:
                    cluster_labels[n_idx] = cluster_id
                i += 1

        for idx in range(n_points):
            if visited[idx]:
                continue
            visited[idx] = True
            neighbors = region_query(idx)
            if len(neighbors) < self.min_samples:
                continue  # 標記為雜訊
            expand_cluster(idx, neighbors, cluster_id)
            cluster_id += 1

        self.labels_ = cluster_labels
        result = labels.copy()
        result[unknown_mask] = cluster_labels + 100  # 分群結果從100開始
        return result
    
    def get_clusters_count(self):
        ''' 返回分群數量 '''
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")
        return len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
