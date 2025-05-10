import pandas as pd
import numpy as np
from typing import Literal

def impute_missing_values(dataset_path: pd.DataFrame,
                          method: Literal['mean', 'medium', 'mode', 'KNN'] = 'mean') -> pd.DataFrame:
    
    match method:
        case 'mean':
            # Impute missing values with the mean of each column
            dataset_path.fillna(dataset_path.mean(), inplace=True)
        case 'medium':
            # Impute missing values with the median of each column
            dataset_path.fillna(dataset_path.median(), inplace=True)
        case 'mode':
            # Impute missing values with the mode of each column
            dataset_path.fillna(dataset_path.mode().iloc[0], inplace=True)
        case 'KNN':
            # Impute missing values using KNN imputation
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            dataset_path = pd.DataFrame(imputer.fit_transform(dataset_path), columns=dataset_path.columns)
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