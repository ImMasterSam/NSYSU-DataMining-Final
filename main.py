import pandas as pd
import numpy as np

from Classifiers.classifier import Classifier
from Classifiers.KNN import *
from Classifiers.RandomF import *
from Classifiers.Neural import *
from Classifiers.SVM import *

from observation.count_missing import count_missing_values
from tools.Imputation import impute_missing_values

models = ('K Nearest Neighbors', 'Linear SVM', 'Neural Network', 'Random Forest', 'Linear SVM', 'Kernel SVM (poly)', 'Kernel SVM (rbf)', 'Kernel SVM (sigmoid)')
model_options = {'K Nearest Neighbors' : KNNClassifier(k = 3, normDistance = 2),
                 'Neural Network' : NeuralNetClassifier(n_hidden = 10, learning_rate = 0.001, n_iters = 1000),
                 'Random Forest' : RandomForestClassifier(n_estimators = 10, max_depth = 10, min_samples_split = 2),
                 'Linear SVM' : SVMClassifier(kernel = 'linear'),
                 'Kernel SVM (linear)' : SVMClassifier(kernel = 'linear'),
                 'Kernel SVM (poly)' : SVMClassifier(kernel = 'poly'),
                 'Kernel SVM (rbf)' : SVMClassifier(kernel = 'rbf'),
                 'Kernel SVM (sigmoid)' : SVMClassifier(kernel = 'sigmoid')}

datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

if __name__ == "__main__":

    for dataset in datasets:

        print(f'\n ----------    {dataset}     --------------\n')

        # Handling Train Data

        train_data_path = f"./dataset/{dataset}/train_data.csv"
        train_label_path = f"./dataset/{dataset}/train_label.csv"

        x_train = pd.read_csv(train_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
        if dataset == 'gene expression cancer RNA-Seq Data Set':
            x_train = x_train.set_index('id')
        y_train = pd.read_csv(train_label_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
        
        impute_missing_values(x_train, 'mean')
        count_missing_values(x_train, True)
        
        # Handling Test Data

        test_data_path = f"./dataset/{dataset}/test_data.csv"
        test_label_path = f"./dataset/{dataset}/test_label.csv"

        x_test = pd.read_csv(test_data_path, header = None if dataset == 'Arrhythmia Data Set' else 0)
        if dataset == 'gene expression cancer RNA-Seq Data Set':
            x_test = x_test.set_index('id')
        y_test = pd.read_csv(test_label_path, header = None if dataset == 'Arrhythmia Data Set' else 0)

        impute_missing_values(x_test, 'mean')
        count_missing_values(x_test, True)

        for model_name in models:

                model: Classifier = model_options[model_name]           # 建立模型
                model.fit(x_train, y_train)                             # 訓練模型
                model.score(x_test, y_test, output = True)              # 測試模型
                
