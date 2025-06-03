from sys import argv
import pandas as pd
import numpy as np
import json

from args import *
from Classifiers.classifier import Classifier
from Clusters.KMeans import KMeansCluster

from tools.Hyperparameter import hyperparameter_tuning
from tools.DataHandler import data_preprocessB
from tools.makepicture import * #畫圖

KMeans = KMeansCluster(n_clusters = 2, max_iter = 300, tol = 1e-4)

def testB_main():
      
    dataset = 'gene expression cancer RNA-Seq Data Set'  # Default dataset

    print(f'\n ----------    {dataset}     --------------\n')

    # Handling Train Data

    print("Loading Train Data...")
    train_data_path = f"./dataset/{dataset}/train_data.csv"
    train_label_path = f"./dataset/{dataset}/train_label.csv"

    x_train = pd.read_csv(train_data_path, header = 0).set_index('id')
    y_train = pd.read_csv(train_label_path, header = 0).set_index('id')
    print("X_train shape:", x_train.shape, ", Y_train shape:", y_train.shape)
    
    # Handling Test Data

    print("Loading Test Data...")
    test_data_path = f"./dataset/{dataset}/test_data.csv"
    test_label_path = f"./dataset/{dataset}/test_label.csv"

    x_test = pd.read_csv(test_data_path, header = 0).set_index('id')
    y_test = pd.read_csv(test_label_path, header = 0).set_index('id')
    print("X_test shape:", x_test.shape, ", Y_test shape:", y_test.shape)

    # 資料預處理
    data_preprocessB(x_train, x_test)
    print("X_train shape:", x_train.shape, ", X_test shape:", x_test.shape)
    print("Data preprocessing completed.\n")

    # Encode labels
    print("Encoding labels...")
    train_unique_labels = list(y_train['Class'].unique())
    test_unique_labels = [label for label in y_test['Class'].unique() if label not in train_unique_labels]
    all_labels = train_unique_labels + test_unique_labels
    label_mapping = {label: idx + 1 for idx, label in enumerate(all_labels)}
    y_train['Class'] = y_train['Class'].map(label_mapping)
    y_test['Class'] = y_test['Class'].map(label_mapping)

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

            model: Classifier = model_options[model_name]                                      # 建立模型
            model.fit(x_train, y_train)                                                        # 訓練模型
            model.score(x_test, y_test, output = True)                                         # 測試模型   

            y_classified = model.predict(x_test)                                               # 預測結果
            best_score = 0.0
            for k in range(2, 5):
                KMeans = KMeansCluster(n_clusters = k, max_iter = 300, tol = 1e-4)
                acc = KMeans.score(x_test, y_classified, y_train, y_test, output = False)      # KMeans 分群 
                # print(f"{model_name} KMeans Score for k={k}: {acc * 100:.2f} %")
                best_score = max(best_score, acc)
            
            print(f"{model_name} KMeans Score: {best_score * 100:.2f} %\n")

            # plot_feature_scatter_double(
            #     x_test,
            #     y_classified,
            #     title_suffix="_Test_Compare",
            #     subfolder=f"trainB/{model_name}"
            # )

if __name__ == "__main__":

    testB_main()      
