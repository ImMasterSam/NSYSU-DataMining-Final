import pandas as pd
import numpy as np

from args import *
from Classifiers.classifier import Classifier
from Clusters.KMeans import KMeansCluster

from observation.count_missing import count_missing_values
from tools.Imputation import impute_missing_values

KMeans = KMeansCluster(n_clusters = 5, max_iter = 300, tol = 1e-4)

def testA_main():

    dataset = 'Arrhythmia Data Set'  # Default dataset

    print(f'\n ----------    {dataset}     --------------\n')

    # Handling Train Data

    print("Handling Train Data...")
    train_data_path = f"./dataset/{dataset}/train_data.csv"
    train_label_path = f"./dataset/{dataset}/train_label.csv"

    x_train = pd.read_csv(train_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
    if dataset == 'gene expression cancer RNA-Seq Data Set':
        x_train = x_train.set_index('id')
    y_train = pd.read_csv(train_label_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
    print("X_train shape:", x_train.shape, ", Y_train shape:", y_train.shape)
    
    print("Imputing missing values in train data...")
    impute_missing_values(x_train, 'mean')
    count_missing_values(x_train, True)
    
    # Handling Test Data

    print("Handling Test Data...")
    test_data_path = f"./dataset/{dataset}/test_data.csv"
    test_label_path = f"./dataset/{dataset}/test_label.csv"

    x_test = pd.read_csv(test_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
    if dataset == 'gene expression cancer RNA-Seq Data Set':
        x_test = x_test.set_index('id')
    y_test = pd.read_csv(test_label_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
    print("X_test shape:", x_test.shape, ", Y_test shape:", y_test.shape)
    
    print("Imputing missing values in test data...")
    impute_missing_values(x_test, 'mean')
    count_missing_values(x_test, True)

    for model_name in models:

            model: Classifier = model_options[model_name]           # 建立模型
            model.fit(x_train, y_train)                             # 訓練模型
            model.score(x_test, y_test, output = True)              # 測試模型   

            y_classified = model.predict(x_test)                    # 預測結果
            y_predict = KMeans.fit_predict(x_test, y_classified)    # KMeans 分群 

            print(y_predict)


if __name__ == "__main__":

    testA_main()
                
