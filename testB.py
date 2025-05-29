import pandas as pd
import numpy as np

from Classifiers.classifier import Classifier
from args import *

from observation.count_missing import count_missing_values
from tools.Imputation import impute_missing_values

def testB_main():
      
    dataset = 'gene expression cancer RNA-Seq Data Set'  # Default dataset

    print(f'\n ----------    {dataset}     --------------\n')

    # Handling Train Data

    print("Handling Train Data...")
    train_data_path = f"./dataset/{dataset}/train_data.csv"
    train_label_path = f"./dataset/{dataset}/train_label.csv"

    x_train = pd.read_csv(train_data_path, header = 0).set_index('id')
    y_train = pd.read_csv(train_label_path, header = 0).set_index('id')
    print("X_train shape:", x_train.shape, ", Y_train shape:", y_train.shape)
    
    print("Imputing missing values in train data...")
    impute_missing_values(x_train, 'mean')
    count_missing_values(x_train, True)
    
    # Handling Test Data

    print("Handling Test Data...")
    test_data_path = f"./dataset/{dataset}/test_data.csv"
    test_label_path = f"./dataset/{dataset}/test_label.csv"

    x_test = pd.read_csv(test_data_path, header = 0).set_index('id')
    y_test = pd.read_csv(test_label_path, header = 0).set_index('id')
    print("X_test shape:", x_test.shape, ", Y_test shape:", y_test.shape)
    
    print("Imputing missing values in test data...")
    impute_missing_values(x_test, 'mean')
    count_missing_values(x_test, True)

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


if __name__ == "__main__":

    testB_main()      
