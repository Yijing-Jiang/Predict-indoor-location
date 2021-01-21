import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

(label_train, data_train_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Train1.csv")
(label_test, data_test_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Test1.csv")
(mean_train, std_train, data_train_norm, data_test_norm) = util.standardized_data (data_train_un, data_test_un)

knf = KNeighborsClassifier(n_neighbors=1)

# unnormalized:
(mean_un, std_un) = util.doCrossOnClf(5, data_train_un, label_train, knf)
print(mean_un, std_un)
(train_acc, train_matrix, test_acc, test_matrix) = util.result(knf, data_train_un, label_train, data_test_un, label_test)
print(train_acc)
print(train_matrix)
print(test_acc)
print(test_matrix)
# normalized: 
(mean_norm, std_norm) = util.doCrossOnClf(5, data_train_norm, label_train, knf)
print(mean_norm, std_norm)
# basic feature selection: 
(set, mean, std) = util.doCrossOnFeature(5, data_train_norm, label_train, knf)
print(set, mean, std)
#PCA feature selection(use normalized data): 
(ratio, pca_train, pca_test) = util.doPCA(7, data_train_norm, data_test_norm)
(mean_pca, std_pca) = util.doCrossOnClf(5, pca_train, label_train, knf)
print(mean_pca, std_pca)
print(ratio)
#LDA feature selection(use normalized data): 
(lda_train, lda_test) = util.doLDA(3, data_train_norm, label_train, data_test_norm)
(mean_lda, std_lda) = util.doCrossOnClf(5, lda_train, label_train, knf)
print(mean_lda, std_lda)
