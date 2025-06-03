# 資料探勘 Data Mining - 第三組

## 目的說明
### 資料集
- **Arrhythmia 資料集**: 特徵包含心律相關檢測資料，目的是要預測病人是否會發生心臟相關疾病。
- **Gene Expression Canver RNA-Seq 資料集**: 特徵為基因序列，目的是要預測病人是否會發生癌症。

### 方法
- 採用「**先分類後分群**」方法
    1. 將分類器較為不確定的樣本點分為未知類。
    2. 使用分群方法將未知類中不熟悉的樣本點進行分群並加入分類結果中。

## 演算法
### 分類器 Classifier
- **KNN**: K-Nearest Neighbors，使用距離度量來進行分類。
- **Neural Network**: 神經網路，使用多層感知器進行分類。
- **Random Forest**: 隨機森林，使用多棵決策樹進行分類。
- **SVM**: Support Vector Machine，使用超平面來進行分類。
    - 一共有 4 種 SVM Kernel:  
        `Linear`、 `Polynomial`、 `RBF`、 `Sigmoid`。

### 分群器 Clustering
- **K-Means**: K-均值分群，將資料點分為 K 個群集。

## 實作方法
### Arrhythmia 資料集
#### 1. 資料前處理: 過濾特徵
- 移除資料過少的特徵: 移除樣本數小於 5 的特徵。
- 過濾常數特徵: 移除所有值都相同的特徵。
- 過濾高度相關特徵: 移除相關係數大於 0.8 的特徵。

#### 2. 資料前處理: 補足缺失值
- 使用平均值 (Mean) 補足缺失值。

#### 3. 資料前處理: 資料過採樣 
因為在 Arrhythmia 資料集中，少數類別(有疾病)的樣本數非常少，因此需要進行資料過採樣。
- 使用 **SMOTE** (Synthetic Minority Over-sampling Technique) 進行資料過採樣，增加少數類別的樣本數。

![Arrhythmia SMOTE](/observation/Label_counts_A.png) 

#### 4. 分類器訓練
- 使用 `KNN`、`Neural Network`、`Random Forest`、`SVM` 分類器進行訓練。
- 將機率小於 `0.8` 的樣本點視為未知，否則將將資料標示類別。

#### 5. 分群器訓練
- 使用 `K-Means` 分群器對分類器預測結果進行分群。

### Arrhythmia 資料集 - 結果

### 純分類結果
| 分類器 | 準確度 Accuracy |
|--------|-----------------|
| KNN    | 38.61 %         | 
| Neural Network | 37.34 %         |
| Random Forest | 53.80 %         |
| SVM (Linear) | 48.10 %         | 
| SVM (Polynomial) | 40.51 %         |
| SVM (RBF) | 34.18 %         |
| SVM (Sigmoid) | 41.14 %         |

### 分群後結果
| 分類器 | 分群器 | 群集數量 | 準確度 Accuracy |
|--------|--------|--------|-----|
| KNN    | K-Means | 1   | 25.32 % |
| Neural Network | K-Means | 2   | 32.28 % |
| Random Forest | K-Means | 6   | 32.28 % |
| SVM (Linear) | K-Means | 3   | **39.87 %** |
| SVM (Polynomial) | K-Means | 3   | 36.08 % |
| SVM (RBF) | K-Means | 5   | 32.91 % |
| SVM (Sigmoid) | K-Means | 1   | 32.91 % |

--- 

### Gene Expression Canver RNA-Seq 資料集
#### 1. 資料前處理
- 過濾常數特徵: 移除所有值都相同的特徵。
- 使用 **SMOTE** (Synthetic Minority Over-sampling Technique) 進行資料過採樣，增加少數類別的樣本數。

![Canver RNA-Seq SMOTE](/observation/Label_counts_B.png) 

#### 2. 分類器訓練
- 使用 `KNN`、`Neural Network`、`Random Forest`、`SVM` 分類器進行訓練。
- 將機率小於 `0.8` 的樣本點視為未知，否則將將資料標示類別。

#### 3. 分群器訓練
- 使用 `K-Means` 分群器對分類器預測結果進行分群。

### Gene Expression Canver RNA-Seq 資料集 - 結果

### 純分類結果
| 分類器 | 準確度 Accuracy |
|--------|-----------------|
| KNN    | 53 %         | 
| Neural Network | 47 %         |
| Random Forest | 94 %         |
| SVM (Linear) | 55 %         | 
| SVM (Polynomial) | 65 %         |
| SVM (RBF) | 72 %         |
| SVM (Sigmoid) | 57 %         |

### 分群後結果
| 分類器 | 分群器 | 群集數量 | 準確度 Accuracy |
|--------|--------|--------|-----|
| KNN    | K-Means | 2   | 52.41 % |
| Neural Network | K-Means | 2   | 46.99 % |
| Random Forest | K-Means | 2   | 93.98 % |
| SVM (Linear) | K-Means | 2   | 54.82 % |
| SVM (Polynomial) | K-Means | 2   | 65.06 % |
| SVM (RBF) | K-Means | 2   | 71.99 % |
| SVM (Sigmoid) | K-Means | 2   | 56.02 % |


## 使用說明
#### 測試兩個資料集
```sh
$ python main.py
```

#### 測試 Arrhythmia 資料集
```sh
$ python testA.py
```

#### 測試 Gene Expression Canver RNA-Seq 資料集
```sh
$ python testB.py
```

## 組員與連結

### Streamlit 連結
Not yet : )   
可能沒有了 : (

### 組員:
- __B123040049劉育希__
-   B123040015陳進發
-   B123040044侯廷翰
-   B123040045林伯儒
-   B123245011楊鎧榤