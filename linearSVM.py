from sklearn.svm import SVC
import util
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

(label_train, data_train_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Train1.csv")
(label_test, data_test_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Test1.csv")
(mean_train, std_train, data_train_norm, data_test_norm) = util.standardized_data (data_train_un, data_test_un)

# basic for SVC
lsvc = SVC(C = 0.99, kernel = 'linear', probability=True)
# unnormalized: 
(mean_un, std_un) = util.doCrossOnClf(5, data_train_un, label_train, lsvc)
print(mean_un, std_un)
# normalized: 
(mean_norm, std_norm) = util.doCrossOnClf(5, data_train_norm, label_train, lsvc)
print(mean_norm, std_norm)
(train_acc, train_matrix, test_acc, test_matrix) = util.result(lsvc, data_train_norm, label_train, data_test_norm, label_test)
print(train_acc)
print(train_matrix)
print(test_acc)
print(test_matrix)
# basic feature selection: 
(set, mean, std) = util.doCrossOnFeature(5, data_train_norm, label_train, lsvc)
print(set, mean, std)
# PCA feature selection(use normalized data):
(ratio, pca_train, pca_test) = util.doPCA(4, data_train_norm, data_test_norm)
(mean_pca, std_pca) = util.doCrossOnClf(5, pca_train, label_train, lsvc)
print(mean_pca, std_pca)
print(ratio)
# LDA feature selection(use normaliezd data): 
(lda_train, lda_test) = util.doLDA(3, data_train_norm, label_train, data_test_norm)
(mean_lda, std_lda) = util.doCrossOnClf(5, lda_train, label_train, lsvc)
print(mean_lda, std_lda)
'''
# find best linear C in normalized data/ PCA with C = 4/ LDA with C = 3
#(ratio, pca_train, pca_test) = util.doPCA(4, data_train_norm, data_test_norm)
#(lda_train, lda_test) = util.doLDA(3, data_train_norm, label_train, data_test_norm)
# linear kernel: C = np.arange(0.05,1,0.02)
# rbf kernel:
num = 9
C = np.logspace(-2,2,num) 
gm = np.logspace(-1,1,num)
mean = np.zeros((num,num))
std = np.zeros((num, num))
for i in range (num): #gamma
    for j in range(num): #C 
        svc = SVC(C = C[j], gamma = gm[i],kernel = 'linear')
        # 'linear', 'poly', 'rbf'
        (mean_c, std_c) = util.doCrossOnClf(1, data_train_norm, label_train, svc)
        # data_train_norm/ pca_train/ lda_train
        mean[i][j] = mean_c
        std[i][j] = std_c
ACC_DEV = mean - std
best_index = np.unravel_index(np.argmax(ACC_DEV, axis=None), ACC_DEV.shape) #convert flat index into matrix index
best_pair = [gm[best_index[0]],C[best_index[1]]]
pair_ACC = mean[best_index[0]][best_index[1]]
pair_DEV = std[best_index[0]][best_index[1]]
print(best_pair, pair_ACC, pair_DEV)
'''
