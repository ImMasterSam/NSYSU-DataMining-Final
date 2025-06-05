import sys
import pandas as pd
import numpy as np
from itertools import product
import json
from tqdm import tqdm

from args import *
from Classifiers.classifier import Classifier
from Clusters.KMeans import KMeansCluster
from Clusters.DBScan import DBScanCluster

from tools.Hyperparameter import hyperparameter_tuning
from tools.DataHandler import data_preprocessA
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

    x_train = pd.read_csv(train_data_path, header=None)
    y_train = pd.read_csv(train_label_path, header=None)
    print("X_train shape:", x_train.shape, ", Y_train shape:", y_train.shape)

    # Read Test Data
    print("Loading Test Data...")
    test_data_path = f"./dataset/{dataset_name}/test_data.csv"
    test_label_path = f"./dataset/{dataset_name}/test_label.csv"

    x_test = pd.read_csv(test_data_path, header=None)
    y_test = pd.read_csv(test_label_path, header=None)
    print("X_test shape:", x_test.shape, ", Y_test shape:", y_test.shape)

def load_models_params(params_filepath):
    """Load model parameters from a JSON file."""

    global x_train, y_train, x_test, y_test

    if len(sys.argv) >= 2 and ('-t' in sys.argv[1:]):
        x_train_preprocessed, y_train_preprocessed, x_test_preprocessed = data_preprocessA(x_train, y_train, x_test)
        model_options = hyperparameter_tuning(x_train_preprocessed, y_train_preprocessed, 5, params_filepath)
    else:
        try:
            models_params = json.load(open(params_filepath, 'r'))
            model_options = {model_name: empty_models[model_name].set_params(**params)
                            for model_name, params in models_params.items()}
            print(f"\n--- 參數設定 ---")
            for model_name, params in models_params.items():
                print(f"{model_name}: {params}")
            print(f"----------------\n")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {params_filepath} not found. Please run with '-t' to generate it.")
        
    return model_options

def testA_main():

    global x_train, y_train, x_test, y_test

    dataset = 'Arrhythmia Data Set'  # Default dataset

    print(f'\n ----------    {dataset}     --------------\n')

    # 載入原始資料
    read_dataset(dataset)

    # 模型參數設定
    params_filepath = 'models_params_A.json'
    model_options = load_models_params(params_filepath)

    result_df = pd.DataFrame({
        'Classification': pd.Series(dtype='str'),
        'Clustering': pd.Series(dtype='str'),
        'Classifier_Accuracy': pd.Series(dtype='float'),
        'Clustering_Accuracy': pd.Series(dtype='float'),
        'constant_threshold': pd.Series(dtype='float'),
        'correlation_threshold': pd.Series(dtype='float'),
        'resampling': pd.Series(dtype='str'),
        'feature_selection_k': pd.Series(dtype='int')
    })
    
    param_iter = product(constant_threshold_params, correlation_threshold_params, resampling_methods, feature_selection_k)
    total_iter = len(constant_threshold_params)*len(correlation_threshold_params)*len(resampling_methods)*len(feature_selection_k)
    for constant_threshold, correlation_threshold, resampling, selection_k in tqdm(param_iter, total=total_iter, desc="Grid Search Progress"):

        # print(f"\n===== constant_threshold = {constant_threshold}, correlation_threshold = {correlation_threshold}, resampling = {resampling}, feature_selection_k = {k} =====")

        # 資料預處理
        x_train_preprocessed, y_train_preprocessed, x_test_preprocessed = data_preprocessA(x_train,
                                                                                           y_train,
                                                                                           x_test,
                                                                                           constant_threshold = constant_threshold,  # 常數特徵的閾值
                                                                                           correlation_threshold = correlation_threshold,  # 高度相關特徵的閾值
                                                                                           resampling = resampling,  # 是否進行重採樣
                                                                                           feature_selection = selection_k)  # 資料預處理
            
        for model_name in models:

            classifier_accs = []
            kmeans_accs = []
            dbscan_accs = []

            for _ in range(repeat_times):
            
                # ===== Classification =====

                model: Classifier = model_options[model_name]                                                   # 建立模型
                model.fit(x_train_preprocessed, y_train_preprocessed)                                           # 訓練模型
                report = model.analysis(x_test_preprocessed, y_test, y_train_preprocessed, output = 0)          # 測試模型   

                y_classified = model.predict(x_test_preprocessed)                                               # 預測結果

                # ===== Clustering =====

                #  KMeans 分群
                best_score = 0.0
                best_k = 0
                for k in range(1, 10):
                    KMeans = KMeansCluster(n_clusters = k, max_iter = 300, tol = 1e-4)
                    acc = KMeans.score(x_test_preprocessed, y_classified, y_train_preprocessed, y_test, output = False)       # KMeans 分群 

                    if acc > best_score:
                        best_k = k
                        best_score = acc

                
                # print(f"{model_name} -> KMeans[{best_k}] Score: {best_score * 100:.2f} %")

                # DBScan 分群
                DBScan = DBScanCluster(eps=3, min_samples=2)
                acc = DBScan.score(x_test_preprocessed, y_classified, y_train_preprocessed, y_test, output = False)
                # print(f"{model_name} -> DBScan Score: {acc * 100:.2f} %\n")

                
                # 新增分數
                classifier_accs.append(report['accuracy'])
                kmeans_accs.append(best_score)
                dbscan_accs.append(acc)

            # 計算平均分數
            classifier_acc = np.mean(classifier_accs)
            kmeans_acc = np.mean(kmeans_accs)
            dbscan_acc = np.mean(dbscan_accs)

            # 儲存結果到 DataFrame
            result_df.loc[len(result_df)] = {
                'Classification': model_name,
                'Clustering': 'KMeans',
                'Classifier_Accuracy': classifier_acc,
                'Clustering_Accuracy': kmeans_acc,
                'constant_threshold': constant_threshold,
                'correlation_threshold': correlation_threshold,
                'resampling': resampling,
                'feature_selection_k': selection_k
            }

            result_df.loc[len(result_df)] = {
                'Classification': model_name,
                'Clustering': 'DBScan',
                'Classifier_Accuracy': classifier_acc,
                'Clustering_Accuracy': dbscan_acc,
                'constant_threshold': constant_threshold,
                'correlation_threshold': correlation_threshold,
                'resampling': resampling,
                'feature_selection_k': selection_k
            }
    
    # 儲存結果到 CSV 檔案
    result_df.to_csv('testA_results.csv', index=False)

if __name__ == "__main__":

    if len(sys.argv) >= 2 and ('-l' in sys.argv[1:]):
        sys.stdout = open('output_A.log', 'w+', encoding='utf-8')

    testA_main()

