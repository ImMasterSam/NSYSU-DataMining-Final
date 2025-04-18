import pandas as pd
import numpy as np

train_data_path = 'dataset/Arrhythmia Data Set/train_data.csv'

train_data = pd.read_csv(train_data_path)
print(train_data.shape)

train_data_path = 'dataset/gene expression cancer RNA-Seq Data Set/train_data.csv'

train_data = pd.read_csv(train_data_path)
print(train_data.shape)