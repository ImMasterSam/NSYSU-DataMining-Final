from Classifiers.KNN import KNNClassifier
from Classifiers.RandomF import RandomForestClassifier
from Classifiers.Neural import NeuralNetClassifier
from Classifiers.SVM import SVMClassifier

models = ('K Nearest Neighbors', 'Neural Network', 'Random Forest', 'Kernel SVM (linear)', 'Kernel SVM (poly)', 'Kernel SVM (rbf)', 'Kernel SVM (sigmoid)')

model_options = {'K Nearest Neighbors' : KNNClassifier(k = 5, normDistance = 2, proba = True, threshold = 0.8),
                 'Neural Network' : NeuralNetClassifier(n_hidden = 50, learning_rate = 0.001, n_iters = 2000, proba = True, threshold = 0.6),
                 'Random Forest' : RandomForestClassifier(n_estimators = 100, max_depth = 20, min_samples_split = 5, proba = True, threshold = 0.6),
                 'Kernel SVM (linear)' : SVMClassifier(kernel = 'linear', proba = True, threshold = 0.6),
                 'Kernel SVM (poly)' : SVMClassifier(kernel = 'poly', proba = True, threshold = 0.6),
                 'Kernel SVM (rbf)' : SVMClassifier(kernel = 'rbf', proba = True, threshold = 0.6),
                 'Kernel SVM (sigmoid)' : SVMClassifier(kernel = 'sigmoid', proba = True, threshold = 0.6)}
