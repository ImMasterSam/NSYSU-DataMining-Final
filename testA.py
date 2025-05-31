import pandas as pd
import numpy as np

from args import *
from Classifiers.classifier import Classifier
from Clusters.KMeans import KMeansCluster

from tools.DataHandler import data_preprocessA
from tools.makepicture import * #畫圖

KMeans = KMeansCluster(n_clusters = 5, max_iter = 300, tol = 1e-4)

def testA_main():

    dataset = 'Arrhythmia Data Set'  # Default dataset

    print(f'\n ----------    {dataset}     --------------\n')

    # Handling Train Data

    print("Loading Train Data...")
    train_data_path = f"./dataset/{dataset}/train_data.csv"
    train_label_path = f"./dataset/{dataset}/train_label.csv"

    x_train = pd.read_csv(train_data_path, header = None)
    y_train = pd.read_csv(train_label_path, header = None)
    print("X_train shape:", x_train.shape, ", Y_train shape:", y_train.shape)
    
    # Handling Test Data

    print("Loading Test Data...")
    test_data_path = f"./dataset/{dataset}/test_data.csv"
    test_label_path = f"./dataset/{dataset}/test_label.csv"

    x_test = pd.read_csv(test_data_path, header = None)
    y_test = pd.read_csv(test_label_path, header = None)
    print("X_test shape:", x_test.shape, ", Y_test shape:", y_test.shape)
    
    # 資料預處理
    data_preprocessA(x_train, y_train, x_test)
    print("X_train shape:", x_train.shape, ", X_test shape:", x_test.shape)
    print("Data preprocessing completed.\n")

    

    for model_name in models:

            model: Classifier = model_options[model_name]           # 建立模型
            model.fit(x_train, y_train)                             # 訓練模型
            model.score(x_test, y_test, output = True)              # 測試模型   

            y_classified = model.predict(x_test)                    # 預測結果
            KMeans.score(x_test, y_classified, y_test, True)        # KMeans 分群 
        
            # 預測訓練資料
            y_train_pred = model.predict(x_train)

            # 畫訓練資料的前處理與分類結果
            plot_feature_scatter_double(
                x_train,
                y_train_pred,  # 這裡用訓練資料的分類結果
                title_suffix="_Train_Compare",
                subfolder=f"trainA/{model_name}"
            )

if __name__ == "__main__":

    testA_main()
                
