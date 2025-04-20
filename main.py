import pandas as pd
import numpy as np

from models.model import Classifier
from models.KNN import *
from models.SVM import *
from models.RandomF import *
from models.Neural import *
from models.SVMK import *

models = ('K Nearest Neighbors', 'Linear SVM', 'Neural Network', 'Random Forest', 'Kernel SVM')
model_options = {'K Nearest Neighbors' : KNNClassifier(k = 21, normDistance = 2, normalize = True),
                 'Linear SVM' : SVMClassifier(learning_rate = 0.001, n_iters = 1000),
                 'Neural Network' : NeuralNetClassifier(n_hidden = 10, learning_rate = 0.001, n_iters = 1000, normalize = True),
                 'Random Forest' : RandomForestClassifier(n_estimators = 10, max_depth = 10, min_samples_split = 2, normalize = True),
                 'Kernel SVM' : SVMClassifierWithKernel(kernel = "rbf", C = 3, gamma = 0.2, n_iters = 1000, normalize = True),}

datasets = ['Arrhythmia Data Set', 'gene expression cancer RNA-Seq Data Set']

for dataset in datasets:

    print(f'\n ----------    {dataset}     --------------\n')

    train_data_path = f"./dataset/{dataset}/train_data.csv"
    train_label_path = f"./dataset/{dataset}/train_label.csv"

    x_train = pd.read_csv(train_data_path)
    y_train = pd.read_csv(train_label_path)

    test_data_path = f"./dataset/{dataset}/test_data.csv"
    test_label_path = f"./dataset/{dataset}/test_label.csv"

    x_test = pd.read_csv(test_data_path)
    y_test = pd.read_csv(test_label_path)

    for model_name in models:

        model: Classifier = model_options[model_name]           # 建立模型
        model.fit(x_train,  y_train)                            # 訓練模型
        model.analysis(x_test, y_test, output = True)           # 測試模型
