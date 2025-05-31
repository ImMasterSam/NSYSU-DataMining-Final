import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.DataHandler import data_preprocessA
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":

    datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

    for dataset in datasets:

        print(f'\n ----------    {dataset}     --------------\n')

        train_data_path = f"./dataset/{dataset}/train_data.csv"
        train_label_path = f"./dataset/{dataset}/train_label.csv"

        test_data_path = f"./dataset/{dataset}/test_data.csv"
        test_label_path = f"./dataset/{dataset}/test_label.csv"

        if dataset == 'Arrhythmia Data Set':
            
            # 讀取訓練資料
            x_train = pd.read_csv(train_data_path, header = None)
            y_train = pd.read_csv(train_label_path, header = None)
            print("X_train shape:", x_train.shape, ", Y_train shape:", y_train.shape)

            # 讀取測試資料
            x_test = pd.read_csv(test_data_path, header = None)
            y_test = pd.read_csv(test_label_path, header = None)
            print("X_test shape:", x_test.shape, ", Y_test shape:", y_test.shape)

            fig, ax = plt.subplots(1,2)

            # 繪製訓練資料的類別分佈
            value_counts = y_train.iloc[:, 0].value_counts()
            print(value_counts)
            ax[0].pie(value_counts, labels = value_counts.index, autopct='%1.1f%%')
            ax[0].set_title('Arrhythmia')  # 圖標題

            # 資料預處理
            data_preprocessA(x_train, y_train, x_test)

            # 繪製處理後的訓練資料類別分佈)
            value_counts = y_train.iloc[:, 0].value_counts()
            print(value_counts)
            ax[1].pie(value_counts, labels = value_counts.index, autopct='%1.1f%%')
            ax[1].set_title('Arrhythmia (After Resample)')  # 圖標題
            plt.tight_layout()
            plt.legend()

            plt.savefig(f"./observation/Label counts.png", dpi = 600)
            plt.show()

        else:
            pass