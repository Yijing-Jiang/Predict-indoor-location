import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler  #standardize data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations 
from sklearn.decomposition import PCA
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# get data from csv
def csv_to_X_and_labels(filename):
    data = pd.read_csv(filename)
    data_array = np.array(data.values.tolist()) #transform data to array
    label = data_array[:,0]
    X = data_array[:,1:]
    return (label, X)

# standardize the data
def standardized_data (X_train, X_test):
    scaler = StandardScaler() #class
    scaler.fit(X_train) #save the mean and var of train data
    mean_train = scaler.mean_
    std_train = np.sqrt(scaler.var_)

    X_train_standardized = scaler.transform(X_train) #use the saved mean and var to standardize train data
    X_test_standardized = scaler.transform(X_test) #use the saved mean and var(train) to standardize test data
    return (mean_train, std_train, X_train_standardized, X_test_standardized) #return mean and std of train data

# do feature selection on cross validation
def doCrossOnFeature(T, data, label, clf):
    skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=None) 
    all_set = []
    for i in range (1,8):
        comb = combinations(range(7), i) 
        for j in list(comb):
            all_set.append(j)
    ACC = []
    for t in range(T):
        for tr_index, v_index in skf.split(data, label):
            #D'-training data
            train_data = data[tr_index,:]
            train_label = label[tr_index]
            #D'-validation data
            v_data = data[v_index,:]
            v_label = label[v_index]
            acc_set = []
            for set in list(all_set):
                train = train_data[:,set]
                v = v_data[:,set]
                clf.fit(train, train_label)
                label_te_pred = clf.predict(v)
                acc = accuracy_score(v_label, label_te_pred)
                acc_set.append(acc)
            ACC.append(acc_set)
    ACC_set = np.mean(ACC, axis = 0)
    STD_set = np.std(ACC, axis = 0)
    ACC_STD = ACC_set - STD_set
    best = np.argmax(ACC_STD)
    return (all_set[best], ACC_set[best], STD_set[best])
    
# do PCA to original data (fit on training set, transform on both training and test set)
def doPCA(c, train, test):
    pca = PCA(n_components = c)
    pca.fit(train)
    train_PCA = pca.transform(train)
    test_PCA = pca.transform(test)
    return (pca.explained_variance_ratio_, train_PCA, test_PCA)

# do Fisher LDA
def doLDA(c, train, label, test):
    lda = LDA(n_components=c)
    lda.fit(train, label)
    train_LDA = lda.transform(train)
    test_LDA = lda.transform(test)
    return (train_LDA, test_LDA)

# use cross validation on different classifier and different data
def doCrossOnClf(T, data, label, clf):
    skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=None)
    acc = []
    for t in range(T):
        for tr_index, v_index in skf.split(data, label):
            #D'-training data
            train_data = data[tr_index,:]
            train_label = label[tr_index]
            #D'-validation data
            v_data = data[v_index,:]
            v_label = label[v_index]
            clf.fit(train_data, train_label)
            label_te_pred = clf.predict(v_data)
            accuracy = accuracy_score(v_label, label_te_pred)
            acc.append(accuracy)
    return (np.mean(acc), np.std(acc))


def result(clf, train_data, train_label, test_data, test_label):
    clf.fit(train_data, train_label)
    train_pred = clf.predict(train_data)
    train_acc = accuracy_score(train_label, train_pred)
    train_matrix = confusion_matrix(train_label, train_pred)
    test_pred = clf.predict(test_data)
    test_acc = accuracy_score(test_label, test_pred)
    test_matrix = confusion_matrix(test_label, test_pred)

    result_confidece (clf.predict_proba(train_data), train_label, train_pred)
    result_confidece (clf.predict_proba(test_data), test_label, test_pred)
    return (train_acc, train_matrix, test_acc, test_matrix)

# with confidence measure:
def result_confidece (matrix, label, label_pred):
    print("with confidence measure: 0.99")
    max = np.max(matrix, axis = -1)
    thre = 0.9
    under_thre_label = []
    under_thre_label_pred = []
    above_thre_label = []
    above_thre_label_pred = []
    for i in range (len(max)):
        if max[i] > thre:
            above_thre_label.append(label[i])
            above_thre_label_pred.append(label_pred[i])
        else:
            under_thre_label.append(label[i])
            under_thre_label_pred.append(label_pred[i])

    acc_test = accuracy_score(above_thre_label, above_thre_label_pred)
    matrix_test = confusion_matrix(above_thre_label, above_thre_label_pred)
    print(acc_test) 
    print(matrix_test)