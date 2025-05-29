from Classifiers.KNN import KNNClassifier
from Classifiers.RandomF import RandomForestClassifier
from Classifiers.Neural import NeuralNetClassifier
from Classifiers.SVM import SVMClassifier

models = ('K Nearest Neighbors', 'Neural Network', 'Random Forest', 'Kernel SVM (linear)', 'Kernel SVM (poly)', 'Kernel SVM (rbf)', 'Kernel SVM (sigmoid)')

model_options = {'K Nearest Neighbors' : KNNClassifier(k = 3, normDistance = 2, proba = True, threshold = 0.7),
                 'Neural Network' : NeuralNetClassifier(n_hidden = 10, learning_rate = 0.001, n_iters = 1000, proba = True, threshold = 0.6),
                 'Random Forest' : RandomForestClassifier(n_estimators = 10, max_depth = 10, min_samples_split = 2),
                 'Kernel SVM (linear)' : SVMClassifier(kernel = 'linear'),
                 'Kernel SVM (poly)' : SVMClassifier(kernel = 'poly'),
                 'Kernel SVM (rbf)' : SVMClassifier(kernel = 'rbf'),
                 'Kernel SVM (sigmoid)' : SVMClassifier(kernel = 'sigmoid')}
