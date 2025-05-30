import pandas as pd
import numpy as np

from tools.Imputation import count_missing_values, impute_missing_values
from tools.Filters import constant_filter, correlation_filter

def data_preprocess(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame) -> None:
    '''資料預處理函式，檢查並填補缺失值'''

    print("\nData preprocessing started...")

    # 檢查資料集是否有常數特徵
    print("Checking for constant features...")
    constant_filter(train_dataset, test_dataset, 0.95)

    # 檢查資料集是否有高度相關的特徵
    print("Checking for highly correlated features...")
    correlation_filter(train_dataset, test_dataset, 0.9, 'pearson')

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