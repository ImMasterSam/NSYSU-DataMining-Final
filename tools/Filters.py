import pandas as pd
import numpy as np
from typing import Literal

def small_sample_filter(train_dataset: pd.DataFrame,
                       train_label: pd.DataFrame,
                       num_samples: int = 5) -> None:
    '''篩選樣本數量少於指定數量的特徵列，過濾掉樣本數量少於 num_samples 的列。'''

    labels_count = train_label.iloc[:, 0].value_counts()
    small_sample_classes = labels_count[labels_count < num_samples].index.tolist()
    
    remove_idx = train_label.iloc[:, 0].isin(small_sample_classes)
    # 直接 in-place 過濾
    train_dataset.drop(train_dataset.index[remove_idx], inplace=True)
    train_label.drop(train_label.index[remove_idx], inplace=True)


def constant_filter(train_dataset: pd.DataFrame,
                    test_dataset: pd.DataFrame,
                    threshold: float = 0.95) -> None:
    '''篩選具有足夠多樣性的特徵列，過濾掉唯一值比例低於閾值的列。'''
    
    constant_feature = []

    for feature in train_dataset.columns:

        # 計算比率
        predominant = (train_dataset[feature].value_counts() / float(len(train_dataset))).sort_values(ascending=False).values[0]
        
        # 假如大於門檻 加入 list
        if predominant >= threshold:
            constant_feature.append(feature)   

    # 移除半常數特徵
    train_dataset.drop(labels=constant_feature, axis=1, inplace=True)
    test_dataset.drop(labels=constant_feature, axis=1, inplace=True)

def correlation_filter(train_dataset: pd.DataFrame,
                       test_dataset: pd.DataFrame,
                       threshold: float = 0.8,
                       method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> None:
    '''篩選具有足夠多樣性的特徵列，過濾掉相關性高於閾值的列。'''
    
    # 建立相關矩陣
    corr_matrix = train_dataset.corr(method = method)

    # 只看上三角（不含對角線）
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in corr_matrix.columns[np.where((corr_matrix.values > threshold) & upper)[1]]]
                
    train_dataset.drop(labels=to_drop, axis=1, inplace=True)
    test_dataset.drop(labels=to_drop, axis=1, inplace=True)