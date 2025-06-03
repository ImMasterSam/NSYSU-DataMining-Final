import numpy as np

from Classifiers.KNN import KNNClassifier
from Classifiers.RandomF import RandomForestClassifier
from Classifiers.Neural import NeuralNetClassifier
from Classifiers.SVM import SVMClassifier

models = ('K Nearest Neighbors', 'Neural Network', 'Random Forest', 'Kernel SVM (linear)', 'Kernel SVM (poly)', 'Kernel SVM (rbf)', 'Kernel SVM (sigmoid)')

all_param_grid = {
    'K Nearest Neighbors': {
        'normalize': [True, False],
        'threshold': [0.8],  # 機率閾值
        'k': list(range(11, 30, 2)),  # 奇數
        'normDistance': list(range(1, 5)),
        'weights': ['uniform', 'distance']
    },
    'Neural Network': {
        'n_hidden': [10, 20, 50],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_iters': [1000, 2000, 3000],
        'normalize': [True, False],
        'threshold': [0.8]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'normalize': [True, False],
        'class_weight': ['balanced', 'balanced_subsample'],
        'threshold': [0.8]
    },
    'Kernel SVM (linear)': {
        'C': list(np.arange(0.3, 10.1, 0.1)),
        'kernel': ['linear'],
        'normalize': [True, False],
        'class_weight': ['balanced', None],
        'threshold': [0.8]
    },
    'Kernel SVM (poly)': {
        'C': list(np.arange(0.3, 10.1, 0.1)),
        'degree': [2, 3, 4],  # 多項式的次數
        'kernel': ['poly'],
        'normalize': [True, False],
        'class_weight': ['balanced', None],
        'threshold': [0.8]
    },
    'Kernel SVM (rbf)': {
        'C': list(np.arange(0.3, 10.1, 0.1)),
        'kernel': ['rbf'],
        'normalize': [True, False],
        'class_weight': ['balanced', None],
        'threshold': [0.8]
    },
    'Kernel SVM (sigmoid)': {
        'C': list(np.arange(0.3, 10.1, 0.1)),
        'kernel': ['sigmoid'],
        'normalize': [True, False],
        'class_weight': ['balanced', None],
        'threshold': [0.8]
    }
}


empty_models = {'K Nearest Neighbors' : KNNClassifier(proba = True),
                'Neural Network' : NeuralNetClassifier(proba = True),
                'Random Forest' : RandomForestClassifier(proba = True),
                'Kernel SVM (linear)' : SVMClassifier(kernel = 'linear', proba = True),
                'Kernel SVM (poly)' : SVMClassifier(kernel = 'poly', proba = True),
                'Kernel SVM (rbf)' : SVMClassifier(kernel = 'rbf', proba = True),
                'Kernel SVM (sigmoid)' : SVMClassifier(kernel = 'sigmoid', proba = True)}

