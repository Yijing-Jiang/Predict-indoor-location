# Predict-indoor-location
Dataset: 

Indoor Wireless Localization (4 classes and 7 features)

Abstract:

This project is to locate the user by using wifi signal strength information. 

GNB (Gussian Naive Bayes) is chosed as baseline model, and will be compared with linear version and rbf version of SVM (Support Vector Machine), MSE(Mean Square Error), Perceptron Learning and KNN (K Nearest Neighbors) algorithm.

For feature preprocessing, consider normalization method. 

For feature adjustment, consider feature combination, PCA (Principal Components Analysis) and FLD (Fisher’s Linear Discriminant) methods.

For each comparison of methods, models or parameters selection, use cross validation method. 

Final best accuracy on test set is 0.9825, using KNN considering 4 nearest beighbors.
