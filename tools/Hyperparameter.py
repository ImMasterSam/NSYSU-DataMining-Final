import pandas as pd
import numpy as np
import json

from sklearn.model_selection import GridSearchCV

from args import *
from Classifiers.classifier import Classifier

def hyperparameter_tuning(x_train: pd.DataFrame,
                          y_train: pd.DataFrame,
                          cv: int = 5,
                          filepath: str = 'models_params.json') -> dict[str, Classifier]:
    ''' 超參數調整 '''

    models_params = {}
    frine_tune_models = {}
    
    for model_name in models:

        print(f"\n--- 調整 {model_name} 的超參數 ---")
        print("調整中 ...")

        param_grid = all_param_grid[model_name]
        clf = empty_models[model_name]
        grid = GridSearchCV(clf, param_grid, cv=cv, n_jobs = -1, scoring='accuracy')
        grid.fit(x_train, y_train.values.ravel())  # y_train 若是 DataFrame 請加 .values.ravel()

        
        print("最佳參數：", grid.best_params_)
        print("最佳分數：", grid.best_score_)

        models_params[model_name] = grid.best_params_
        frine_tune_models[model_name] = grid.best_estimator_

    print("\n--- 超參數調整完成 ---")
    json.dump(models_params, open(filepath, 'w+'), indent=4)
    return frine_tune_models


