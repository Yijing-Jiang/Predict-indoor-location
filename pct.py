import util
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

(label_train, data_train_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Train1.csv")
(label_test, data_test_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Test1.csv")
(mean_train, std_train, data_train_norm, data_test_norm) = util.standardized_data (data_train_un, data_test_un)

pct = Perceptron(tol=1e-3, random_state=0)
# unnormalized:
(mean_un, std_un) = util.doCrossOnClf(5, data_train_un, label_train, pct)
print(mean_un, std_un)
# normalized:
(mean_norm, std_norm) = util.doCrossOnClf(5, data_train_norm, label_train, pct)
print(mean_norm, std_norm)
# basic feature selection: 
(set, mean, std) = util.doCrossOnFeature(5, data_train_norm, label_train, pct)
print(set, mean, std)
#PCA feature selection(use normalized data):
(ratio, pca_train, pca_test) = util.doPCA(7, data_train_norm, data_test_norm)
(mean_pca, std_pca) = util.doCrossOnClf(5, pca_train, label_train, pct)
print(mean_pca, std_pca)
print(ratio)
(train_acc, train_matrix, test_acc, test_matrix) = util.result(pct, pca_train, label_train, pca_test, label_test)
print(train_acc)
print(train_matrix)
print(test_acc)
print(test_matrix)
#LDA feature selection(use normalized data)
(lda_train, lda_test) = util.doLDA(2, data_train_norm, label_train, data_test_norm)
(mean_lda, std_lda) = util.doCrossOnClf(5, lda_train, label_train, pct)
print(mean_lda, std_lda)
