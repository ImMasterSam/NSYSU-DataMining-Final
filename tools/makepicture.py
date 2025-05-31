import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_scatter_double(x_train, y_pred, title_suffix="", subfolder=""):
    """將 preprocess 後與分類後的 train 資料分別畫在同一張圖的兩個子圖"""
    # 只取前10個特徵避免太多
    if x_train.shape[1] > 10:
        selected_features = x_train.columns[:10].tolist()
    else:
        selected_features = x_train.columns.tolist()

    n = len(selected_features)
    for i in range(n):
        for j in range(i+1, n):
            x_col = selected_features[i]
            y_col = selected_features[j]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 左圖：前處理後的 train
            sns.scatterplot(x=x_train[x_col], y=x_train[y_col], color="gray", alpha=0.7, ax=axes[0])
            axes[0].set_title(f"Preprocessed: {x_col} vs {y_col}")
            axes[0].set_xlabel(x_col)
            axes[0].set_ylabel(y_col)

            # 右圖：分類後的 train（分色）
            sns.scatterplot(x=x_train[x_col], y=x_train[y_col], hue=y_pred, palette="tab10", alpha=0.8, ax=axes[1])
            axes[1].set_title(f"Classified: {x_col} vs {y_col}")
            axes[1].set_xlabel(x_col)
            axes[1].set_ylabel(y_col)
            axes[1].legend(title="Pred", bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.suptitle(f"{x_col} vs {y_col}{title_suffix}", fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # 儲存
            base_dir = 'picture'
            if subfolder:
                save_dir = os.path.join(base_dir, subfolder)
            else:
                save_dir = base_dir
            os.makedirs(save_dir, exist_ok=True)
            filename = f"scatter_double_{x_col}_vs_{y_col}{title_suffix.replace(' ', '_')}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
