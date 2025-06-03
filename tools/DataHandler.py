import pandas as pd
import numpy as np

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from tools.Imputation import count_missing_values, impute_missing_values
from tools.Filters import small_sample_filter, constant_filter, correlation_filter

def data_preprocessA(train_dataset: pd.DataFrame, train_label: pd.DataFrame, test_dataset: pd.DataFrame) -> None:
    '''資料預處理函式，針對 Arrhythmia Data Set 進行處理'''

    print("\nData preprocessing started...")

    # 篩選樣本數量少於指定數量的特徵列
    print("Filtering out features with small sample sizes...")
    small_sample_filter(train_dataset, train_label, 5)

    # 檢查資料集是否有常數特徵
    print("Checking for constant features...")
    constant_filter(train_dataset, test_dataset, 0.7)

    # 檢查資料集是否有高度相關的特徵
    print("Checking for highly correlated features...")
    correlation_filter(train_dataset, test_dataset, 0.8, 'pearson')

    # 檢查資料集是否有缺失值
    if count_missing_values(train_dataset, False).size > 0:
        print("Imputing missing values in train data...")
        impute_missing_values(train_dataset, 'mean')
    if count_missing_values(test_dataset, False).size > 0:
        print("Imputing missing values in test data...")
        impute_missing_values(test_dataset, 'mean')

    # 進行資料過採樣
    print("Over-sampling with SMOTE & Tomek...")
    X_res, y_res = SMOTETomek().fit_resample(train_dataset, train_label)
    # print("Over-sampling with SMOTE...")
    # X_oversample_res, y_oversample_res = SMOTE().fit_resample(train_dataset, train_label)
    # print("Under-sampling with NearMiss...")
    # X_res, y_res = NearMiss(version = 3).fit_resample(X_oversample_res, y_oversample_res)

    # 更新訓練資料集和標籤
    print("Updating train dataset and label with resampled data...")
    train_dataset.drop(train_dataset.index, inplace=True)
    for col in train_dataset.columns:
        train_dataset[col] = X_res[col]
    train_label.drop(train_label.index, inplace=True)
    train_label[train_label.columns[0]] = y_res

def data_preprocessB(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame) -> None:
    '''資料預處理函式，針對 gene expression cancer RNA-Seq Data Set 進行處理'''

    print("\nData preprocessing started...")

    # 檢查資料集是否有常數特徵
    print("Checking for constant features...")
    constant_filter(train_dataset, test_dataset, 0.7)

    # 檢查資料集是否有高度相關的特徵
    # print("Checking for highly correlated features...")
    # correlation_filter(train_dataset, test_dataset, 0.8, 'pearson')

    # 檢查資料集是否有缺失值
    if count_missing_values(train_dataset, False).size > 0:
        print("Imputing missing values in train data...")
        impute_missing_values(train_dataset, 'mean')
    if count_missing_values(test_dataset, False).size > 0:
        print("Imputing missing values in test data...")
        impute_missing_values(test_dataset, 'mean')

if __name__ == "__main__":

    datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

    for dataset in datasets:

        print(f'\n ----------    {dataset}     --------------\n')

        train_data_path = f"./dataset/{dataset}/train_data.csv"
        train_label_path = f"./dataset/{dataset}/train_label.csv"

        x_train = pd.read_csv(train_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
        impute_missing_values(x_train, 'mean')