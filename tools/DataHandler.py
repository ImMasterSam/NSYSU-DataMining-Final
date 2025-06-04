import pandas as pd
import numpy as np

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

from tools.Imputation import count_missing_values, impute_missing_values
from tools.Filters import small_sample_filter, constant_filter, correlation_filter

def data_preprocessA(train_dataset: pd.DataFrame,
                     train_label: pd.DataFrame,
                     test_dataset: pd.DataFrame,
                     output: bool = False,
                     constant_threshold: float | None = 0.7,
                     correlation_threshold: float | None = 0.8,
                     resampling: bool = True,
                     feature_selection: int | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''資料預處理函式，針對 Arrhythmia Data Set 進行處理'''

    if output:
        print("\nData preprocessing started...")

    # 篩選樣本數量少於指定數量的特徵列
    if output:
        print("Filtering out features with small sample sizes...")
    train_dataset, train_label = small_sample_filter(train_dataset, train_label, 5)

    # 檢查資料集是否有常數特徵
    if constant_threshold is not None:
        if output:
            print("Checking for constant features...")
        train_dataset, test_dataset = constant_filter(train_dataset, test_dataset, constant_threshold)

    # 檢查資料集是否有高度相關的特徵
    if correlation_threshold is not None:
        if output:
            print("Checking for highly correlated features...")
        train_dataset, test_dataset = correlation_filter(train_dataset, test_dataset, correlation_threshold, 'pearson')

    # 檢查資料集是否有缺失值
    if count_missing_values(train_dataset, False).size > 0:
        if output:
            print("Imputing missing values in train data...")
        train_dataset = impute_missing_values(train_dataset, 'mean')
    if count_missing_values(test_dataset, False).size > 0:
        if output:
            print("Imputing missing values in test data...")
        test_dataset = impute_missing_values(test_dataset, 'mean')

    # 進行資料過採樣
    if resampling:
        if output:
            print("Resampling with SMOTE & Tomek...")
        X_res, y_res = SMOTETomek().fit_resample(train_dataset, train_label)
        train_dataset = pd.DataFrame(X_res, columns=train_dataset.columns)
        train_label = pd.DataFrame(y_res, columns=train_label.columns)

    # 選擇要保留的特徵數
    if feature_selection is not None:
        select_k = min(feature_selection, train_dataset.shape[1])  # 確保不超過特徵數量
        selection = SelectKBest(mutual_info_classif, k=select_k).fit(train_dataset, train_label.values.ravel())
        features = train_dataset.columns[selection.get_support()]

        if output:
            print(f"Selected top {select_k} features : {features.tolist()}") # 顯示保留的欄位

        train_dataset = train_dataset[features].copy()
        test_dataset = test_dataset[features].copy()

    if output:
        print("Data preprocessing completed.\n")

    return train_dataset, train_label, test_dataset

def data_preprocessB(train_dataset: pd.DataFrame,
                     train_label: pd.DataFrame,
                     test_dataset: pd.DataFrame = None,
                     output: bool = False,
                     constant_threshold: float | None = 0.7,
                     correlation_threshold: float | None = None,
                     resampling: bool = True,
                     feature_selection: int | None = 256
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''資料預處理函式，針對 gene expression cancer RNA-Seq Data Set 進行處理'''

    if output:
        print("\nData preprocessing started...")

    # 檢查資料集是否有常數特徵
    if constant_threshold is not None:
        if output:
            print("Checking for constant features...")
        train_dataset, test_dataset = constant_filter(train_dataset, test_dataset, constant_threshold)

    # 檢查資料集是否有高度相關的特徵
    if correlation_threshold is not None:
        if output:
            print("Checking for highly correlated features...")
        train_dataset, test_dataset = correlation_filter(train_dataset, test_dataset, correlation_threshold, 'pearson')

    # 檢查資料集是否有缺失值
    if count_missing_values(train_dataset, False).size > 0:
        if output:
            print("Imputing missing values in train data...")
        train_dataset = impute_missing_values(train_dataset, 'mean')
    if count_missing_values(test_dataset, False).size > 0:
        if output:
            print("Imputing missing values in test data...")
        test_dataset = impute_missing_values(test_dataset, 'mean')

    # 進行資料過採樣
    if resampling:
        if output:
            print("Resampling with SMOTE & Tomek...")
        X_res, y_res = SMOTETomek().fit_resample(train_dataset, train_label)
        train_dataset = pd.DataFrame(X_res, columns=train_dataset.columns)
        train_label = pd.DataFrame(y_res, columns=train_label.columns)

    # 選擇要保留的特徵數
    if feature_selection is not None:
        select_k = min(feature_selection, train_dataset.shape[1])
        selection = SelectKBest(mutual_info_classif, k=select_k).fit(train_dataset, train_label.values.ravel())
        features = train_dataset.columns[selection.get_support()]
        if output:
            print(f"Selected top {select_k} features : {features.tolist()}")
        train_dataset = train_dataset[features].copy()
        test_dataset = test_dataset[features].copy()

    if output:
        print("Data preprocessing completed.\n")

    return train_dataset, train_label, test_dataset

if __name__ == "__main__":

    datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

    for dataset in datasets:

        print(f'\n ----------    {dataset}     --------------\n')

        train_data_path = f"./dataset/{dataset}/train_data.csv"
        train_label_path = f"./dataset/{dataset}/train_label.csv"

        x_train = pd.read_csv(train_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
        impute_missing_values(x_train, 'mean')