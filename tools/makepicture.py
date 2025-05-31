import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

def plot_classifier_and_cluster(X, y_pred, cluster_labels, title_suffix="", subfolder=""):
    # PCA降維
    if X.shape[1] > 2:
        X_vis = PCA(n_components=2).fit_transform(X)
    else:
        X_vis = X

    # jitter
    jitter_strength = 1
    X_vis = X_vis + np.random.normal(0, jitter_strength, X_vis.shape)

    plt.figure(figsize=(12, 5))

    # 左圖：class 專用顏色
    class_labels = np.unique(y_pred)
    class_colors = ['#B0B0B0'] + list(plt.cm.tab10.colors)  # -1用灰色，其他用tab10
    class_cmap = ListedColormap(class_colors[:len(class_labels)])
    class_color_idx = {label: idx for idx, label in enumerate(class_labels)}
    color_indices1 = [class_color_idx[v] for v in y_pred]

    # 右圖：cluster 專用顏色
    cluster_unique = np.unique(cluster_labels)
    cluster_colors = list(plt.cm.tab20.colors)  # cluster 用 tab20
    cluster_cmap = ListedColormap(cluster_colors[:len(cluster_unique)])
    cluster_color_idx = {label: idx for idx, label in enumerate(cluster_unique)}
    color_indices2 = [cluster_color_idx[v] for v in cluster_labels]

    # 左圖：class
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(
        X_vis[:, 0], X_vis[:, 1],
        c=color_indices1,
        cmap=class_cmap, alpha=0.8, edgecolor='k', s=20, linewidth=0.3
    )
    plt.title(f'Classifier Prediction{title_suffix}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 只顯示 class 有出現過的顏色
    handles1 = []
    for c in class_labels:
        idx = class_color_idx[c]
        handles1.append(plt.Line2D([], [], marker='o', color='w',
                                   markerfacecolor=class_colors[idx], markeredgecolor='k', markersize=8, label=str(c)))
    plt.legend(handles=handles1, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 右圖：cluster
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(
        X_vis[:, 0], X_vis[:, 1],
        c=color_indices2,
        cmap=cluster_cmap, alpha=0.8, edgecolor='k', s=20, linewidth=0.3
    )
    plt.title(f'Cluster Result{title_suffix}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 只顯示 cluster 有出現過的顏色
    handles2 = []
    for c in cluster_unique:
        idx = cluster_color_idx[c]
        handles2.append(plt.Line2D([], [], marker='o', color='w',
                                   markerfacecolor=cluster_colors[idx], markeredgecolor='k', markersize=8, label=str(c)))
    plt.legend(handles=handles2, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # 決定子資料夾
    base_dir = 'picture'
    if subfolder:
        save_dir = os.path.join(base_dir, subfolder)
    else:
        save_dir = base_dir
    os.makedirs(save_dir, exist_ok=True)
    filename = f"classifier_cluster{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()