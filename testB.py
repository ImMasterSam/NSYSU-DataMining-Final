import pandas as pd
import numpy as np

from args import *
from Classifiers.classifier import Classifier
from Clusters.KMeans import KMeansCluster

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
    unique_labels = set(y_train['Class'].unique()) | set(y_test['Class'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_train['Class'] = y_train['Class'].map(label_mapping)
    y_test['Class'] = y_test['Class'].map(label_mapping)

    for model_name in models:

            model: Classifier = model_options[model_name]           # 建立模型
            model.fit(x_train, y_train)                             # 訓練模型
            model.score(x_test, y_test, output = True)              # 測試模型   

            y_classified = model.predict(x_test)                    # 預測結果
            KMeans.score(x_test, y_classified, y_test, True)        # KMeans 分群 

            plot_feature_scatter_double(
                x_test,
                y_classified,
                title_suffix="_Test_Compare",
                subfolder=f"trainB/{model_name}"
            )

if __name__ == "__main__":

    testB_main()      
