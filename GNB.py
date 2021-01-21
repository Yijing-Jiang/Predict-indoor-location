import util
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

(label_train, data_train_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Train1.csv")
(label_test, data_test_un) = util.csv_to_X_and_labels("/Users/mac-pro/Desktop/20SPRING/559/Assignment/location/D_Test1.csv")
(mean_train, std_train, data_train_norm, data_test_norm) = util.standardized_data (data_train_un, data_test_un)

# baseline model
gnb = GaussianNB()
# unnormalized: 
list = [0,1,2,3,4,6,]
(mean_un, std_un) = util.doCrossOnClf(5, data_train_un[:,list], label_train, gnb)
print(mean_un, std_un)
(train_acc, train_matrix, test_acc, test_matrix) = util.result(gnb, data_train_un, label_train, data_test_un, label_test)
print(train_acc)
print(train_matrix)
print(test_acc)
print(test_matrix)
# normalized:
(mean_norm, std_norm) = util.doCrossOnClf(5, data_train_norm, label_train, gnb)
print(mean_norm, std_norm)
# basic feature selection:
(set, mean, std) = util.doCrossOnFeature(5, data_train_un, label_train, gnb)
print(set, mean, std)
#PCA feature selection(use normalized data):  
#    try different C in doPCA(): C = 2,3,4,5,6,7
(ratio, pca_train, pca_test) = util.doPCA(4, data_train_norm, data_test_norm)
(mean_pca, std_pca) = util.doCrossOnClf(5, pca_train, label_train, gnb)
print(mean_pca, std_pca)
print(ratio)
#LDA feature selection(use normalized data): 
#    try different C in doLDA(): C = 1,2,3
(lda_train, lda_test) = util.doLDA(3, data_train_norm, label_train, data_test_norm)
(mean_lda, std_lda) = util.doCrossOnClf(5, lda_train, label_train, gnb)
print(mean_lda, std_lda)
# plot data in 2D space
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g', 'b', 'y']
for i in range(1600):
    ax.scatter(lda_train[i,0], lda_train[i,1], c = colors[label_train[i]-1], s = 50)
ax.grid()
plt.show()
