from sys import argv
import pandas as pd
import numpy as np
import json

from args import *
from Classifiers.classifier import Classifier
from Clusters.KMeans import KMeansCluster

from tools.Hyperparameter import hyperparameter_tuning
from tools.DataHandler import data_preprocessA
from tools.makepicture import * #畫圖

KMeans = KMeansCluster(n_clusters = 7, max_iter = 300, tol = 1e-4)

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

    params_filepath = 'models_params_A.json'
    model_options = {}

    if len(argv) >= 2:
        if argv[1] == '-t':
            model_options = hyperparameter_tuning(x_train, y_train, 2, params_filepath)  # 超參數調整
        else:
            print("Invalid argument. Use '-t' for hyperparameter tuning.")
            return
    else:
        try:
            models_params = json.load(open(params_filepath, 'r'))
            model_options = {model_name: empty_models[model_name].set_params(**params)
                            for model_name, params in models_params.items()}
        except FileNotFoundError:
            raise FileNotFoundError(f"File {params_filepath} not found. Please run with '-t' to generate it.")
    

    for model_name in models:

            model: Classifier = model_options[model_name]                   # 建立模型
            model.fit(x_train, y_train)                                     # 訓練模型
            model.analysis(x_test, y_test, y_train, output = False)         # 測試模型   

            y_classified = model.predict(x_test)                                    # 預測結果
            best_score = 0.0
            for k in range(1, 10):
                KMeans = KMeansCluster(n_clusters = k, max_iter = 300, tol = 1e-4)
                acc = KMeans.score(x_test, y_classified, y_train, y_test, output = False)      # KMeans 分群 
                # print(f"{model_name} KMeans Score for k={k}: {acc * 100:.2f} %")
                best_score = max(best_score, acc)
            
            print(f"{model_name} KMeans Score: {best_score * 100:.2f} %\n")
        
            # # 預測訓練資料
            # y_train_pred = model.predict(x_train)

            # # 畫訓練資料的前處理與分類結果
            # plot_feature_scatter_double(
            #     x_train,
            #     y_train_pred,  # 這裡用訓練資料的分類結果
            #     title_suffix="_Train_Compare",
            #     subfolder=f"trainA/{model_name}"
            # )

if __name__ == "__main__":

    testA_main()
                
