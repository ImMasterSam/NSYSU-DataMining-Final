import numpy as np
import pandas as pd

from Clusters.cluster import Cluster

class KMeansCluster(Cluster):

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        super().__init__(name = 'KMeans Cluster')
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit_predict(self, x_train, labels):
        labels = np.array(labels)
        unknown_mask = (labels == -1)
        x_unknown = x_train[unknown_mask]
        x_unknown = np.array(x_unknown)

        # 若沒有未知資料，直接回傳原標籤
        if len(x_unknown) == 0:
            return labels

        # 初始化 centroids
        indices = np.random.choice(len(x_unknown), self.n_clusters, replace=False)
        centroids = x_unknown[indices]

        for _ in range(self.max_iter):
            # 計算每個點到各中心的距離
            distances = np.linalg.norm(x_unknown[:, None] - centroids[None, :], axis=2)
            cluster_labels = np.argmin(distances, axis=1)
            new_centroids = np.array([x_unknown[cluster_labels == i].mean(axis=0) if np.any(cluster_labels == i) else centroids[i] for i in range(self.n_clusters)])
            # 收斂判斷
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < self.tol):
                break
            centroids = new_centroids
        self.centroids = centroids

        # 將分群結果回填到原 labels
        result = labels.copy()
        result[unknown_mask] = cluster_labels + 100 # 假設分群結果從 100 開始編號，避免與原標籤衝突

        return result
    