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

x_train = pd.DataFrame()
y_train = pd.DataFrame()
x_test = pd.DataFrame()
y_test = pd.DataFrame()

def read_dataset(dataset_name):

    global x_train, y_train, x_test, y_test

    # Read Train Data
    print("Loading Train Data...")
    train_data_path = f"./dataset/{dataset_name}/train_data.csv"
    train_label_path = f"./dataset/{dataset_name}/train_label.csv"

    x_train = pd.read_csv(train_data_path, header=0).set_index('id')
    y_train = pd.read_csv(train_label_path, header=0).set_index('id')
    print("X_train shape:", x_train.shape, ", Y_train shape:", y_train.shape)

    # Read Test Data
    print("Loading Test Data...")
    test_data_path = f"./dataset/{dataset_name}/test_data.csv"
    test_label_path = f"./dataset/{dataset_name}/test_label.csv"

    x_test = pd.read_csv(test_data_path, header=0).set_index('id')
    y_test = pd.read_csv(test_label_path, header=0).set_index('id')
    print("X_test shape:", x_test.shape, ", Y_test shape:", y_test.shape)

def testB_main():

    global x_train, y_train, x_test, y_test
      
    dataset = 'gene expression cancer RNA-Seq Data Set'  # Default dataset

    print(f'\n ----------    {dataset}     --------------\n')

    # Read dataset
    read_dataset(dataset)

    # Encode labels
    print("Encoding labels...")
    train_unique_labels = list(y_train['Class'].unique())
    test_unique_labels = [label for label in y_test['Class'].unique() if label not in train_unique_labels]
    all_labels = train_unique_labels + test_unique_labels
    label_mapping = {label: idx + 1 for idx, label in enumerate(all_labels)}
    y_train['Class'] = y_train['Class'].map(label_mapping)
    y_test['Class'] = y_test['Class'].map(label_mapping)

    # 資料預處理
    data_preprocessB(x_train, y_train, x_test)
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

            model: Classifier = model_options[model_name]                                       # 建立模型
            model.fit(x_train, y_train)                                                         # 訓練模型
            model.analysis(x_test, y_test, y_train, output = True)                              # 測試模型   

            y_classified = model.predict(x_test)                                                # 預測結果
            best_score = 0.0
            best_k = 0
            for k in range(2, 5):
                KMeans = KMeansCluster(n_clusters = k, max_iter = 300, tol = 1e-4)
                acc = KMeans.score(x_test, y_classified, y_train, y_test, output = False)       # KMeans 分群 
                # print(f"{model_name} KMeans Score for k={k}: {acc * 100:.2f} %")
                if acc > best_score:
                    best_k = k
                    best_score = acc
            
            print(f"{model_name} -> KMeans[{best_k}] Score: {best_score * 100:.2f} %\n")

            # plot_feature_scatter_double(
            #     x_test,
            #     y_classified,
            #     title_suffix="_Test_Compare",
            #     subfolder=f"trainB/{model_name}"
            # )

if __name__ == "__main__":

    testB_main()      
