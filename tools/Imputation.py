import pandas as pd
import numpy as np
from typing import Literal

def count_missing_values(dataset_path: pd.DataFrame, output: bool = False) -> pd.DataFrame:
    """
    Count the number of missing values in each column of the dataset.

    Returns:
    pd.DataFrame: DataFrame containing the count of missing values for each column.
    """

    # Count missing values in each column
    missing_counts = dataset_path.isnull().sum()
    missing_counts = pd.DataFrame(missing_counts[missing_counts > 0], columns=['Missing Count'])

    if output:
        if missing_counts.empty:
            print("No missing values found in the dataset.")
        else:
            for index in missing_counts.index:
                print(f"Column {index} has {missing_counts.loc[index, 'Missing Count']: 4d} missing values.")
                print("With unique values:")
                print(dataset_path[index].unique())
    
    return missing_counts

def impute_missing_values(dataset: pd.DataFrame,
                          method: Literal['mean', 'medium', 'mode', 'KNN'] = 'mean') -> pd.DataFrame:
    """
    回傳填補缺失值後的新 DataFrame。
    """
    match method:
        case 'mean':
            # Impute missing values with the mean of each column
            return dataset.fillna(dataset.mean())
        case 'medium':
            # Impute missing values with the median of each column
            return dataset.fillna(dataset.median())
        case 'mode':
            # Impute missing values with the mode of each column
            return dataset.fillna(dataset.mode().iloc[0])
        case 'KNN':
            # Impute missing values using KNN imputation
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            return pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        case _:
            raise ValueError("Invalid method. Choose from 'mean', 'medium', 'mode', or 'KNN'.")
        
if __name__ == "__main__":

    datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

    for dataset in datasets:

        print(f'\n ----------    {dataset}     --------------\n')

        train_data_path = f"./dataset/{dataset}/train_data.csv"
        train_label_path = f"./dataset/{dataset}/train_label.csv"

        x_train = pd.read_csv(train_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
        impute_missing_values(x_train, 'mean')