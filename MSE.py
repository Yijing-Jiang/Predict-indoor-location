import numpy as np
import util
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier  #classifier

class MSE_binary (LinearRegression):
    def __init__ (self):
        super(MSE_binary, self).__init__()
    def predict(self, X):
        thr = 0.475 # if label is 1 and 0
        y = self._decision_function(X) # predicted label = X * w + w0
        len = np.prod(y.shape)
        y_binary = y
        for i in range(len):
            if y[i] >= thr: y_binary[i] = 1
            else: y_binary[i] = 0
        return y_binary

(label_train, data_train_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Train1.csv")
(label_test, data_test_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Test1.csv")
(mean_train, std_train, data_train_norm, data_test_norm) = util.standardized_data (data_train_un, data_test_un)

# MSE
binary_model = MSE_binary()
mse = OneVsRestClassifier(binary_model) 
# nonnormalized: 
(mean_un, std_un) = util.doCrossOnClf(5, data_train_un, label_train, mse)
print(mean_un, std_un)
# normalized data:
(mean_norm, std_norm) = util.doCrossOnClf(5, data_train_norm, label_train, mse)
print(mean_norm, std_norm)
# basic feature selection: 
(set, mean, std) = util.doCrossOnFeature(5, data_train_norm, label_train, mse)
print(set, mean, std)
#PCA feature selection(use normalized data): 
(ratio, pca_train, pca_test) = util.doPCA(4, data_train_norm, data_test_norm)
(mean_pca, std_pca) = util.doCrossOnClf(5, pca_train, label_train, mse)
print(mean_pca, std_pca)
print(ratio)
#LDA feature selection(use normalized data): 
(lda_train, lda_test) = util.doLDA(3, data_train_norm, label_train, data_test_norm)
(mean_lda, std_lda) = util.doCrossOnClf(5, lda_train, label_train, mse)
print(mean_lda, std_lda)
(train_acc, train_matrix, test_acc, test_matrix) = util.result(mse, lda_train, label_train, lda_test, label_test)
print(train_acc)
print(train_matrix)
print(test_acc)
print(test_matrix)