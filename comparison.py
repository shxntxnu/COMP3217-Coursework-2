import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

x = []
y = []

with open("TrainingData.txt") as f:
    lines = f.readlines()
    lines = lines[:]
    for in_ds, item in enumerate(lines):
        lines[in_ds] = lines[in_ds].strip("\n").split(",")
        lines[in_ds] = [float(v) for v in lines[in_ds]]
        x.append(lines[in_ds][0:-1])
        y.append(lines[in_ds][-1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x = np.array(x)
y = np.array(y)
x_train_full = x
y_train_full = y

# Normalising by scaling from 0-1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train_full_2 = scaler.transform(x_train_full)

# KNN
start_KNN = time.time()
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)
y_pred_train = neigh.predict(x_train_full_2)
print("\nAccuracy for KNN on Training Set:", (metrics.accuracy_score(y_train_full, y_pred_train)) * 100, "%")
print("Accuracy for KNN on 20% Testing Set:", (metrics.accuracy_score(y_test, y_pred)) * 100, "%")
end_KNN = time.time()
print("Time Taken: ", end_KNN - start_KNN, "s")

# Decision Tree
start_Dec_Tree = time.time()
tree = DecisionTreeClassifier().fit(x_train, y_train)
print("\nAccuracy for Decision Tree classifier on Training set:", (tree.score(x_train_full_2, y_train_full)) * 100, "%")
print("Accuracy for Decision Tree classifier on 20% Testing set:", (tree.score(x_test, y_test)) * 100, "%")
end_Dec_Tree = time.time()
print("Time Taken: ", end_Dec_Tree - start_Dec_Tree, "s")

# Logistic Regression
start_Log_Reg = time.time()
log = linear_model.LogisticRegression(random_state=2, solver='liblinear', multi_class='auto')
log = log.fit(x_train, y_train)
print("\nAccuracy for Logistic Regression on Training set:", (log.score(x_train_full_2, y_train_full)) * 100, "%")
print("Accuracy for Logistic Regression on 20% Testing set:", (log.score(x_test, y_test)) * 100, "%")
end_Log_Reg = time.time()
print("Time Taken: ", end_Log_Reg - start_Log_Reg, "s")

# Support Vector Machine
start_SVM = time.time()
svm = SVC()
svm.fit(x_train, y_train)
print("\nAccuracy for SVM classifier on Training set:", (svm.score(x_train_full_2, y_train_full)) * 100, "%")
print("Accuracy for SVM classifier on 20% Testing set:", (svm.score(x_test, y_test)) * 100, "%")
end_SVM = time.time()
print("Time Taken: ", end_SVM - start_SVM, "s")

# Linear Discriminant Analysis
start_LDA = time.time()
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
print("\nAccuracy for LDA classifier on Training set:", (lda.score(x_train_full_2, y_train_full)) * 100, "%")
print("Accuracy for LDA classifier on 20% Testing set:", (lda.score(x_test, y_test)) * 100, "%")
end_LDA = time.time()
print("Time Taken: ", end_LDA - start_LDA, "s")

# Gaussian Naive Bayes
start_GNB = time.time()
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print("\nAccuracy for GNB classifier on Training set:", (gnb.score(x_train_full_2, y_train_full)) * 100, "%")
print("Accuracy for GNB classifier on 20% Testing set:", (gnb.score(x_test, y_test)) * 100, "%")
end_GNB = time.time()
print("Time Taken: ", end_GNB - start_GNB, "s")

# Multi Layer Perception
start_MLP = time.time()
reg = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="relu", random_state=1, max_iter=2000).fit(x_train,
                                                                                                          y_train)
y_pred_mlp = reg.predict(x_test)
y_pred_mlp_train = reg.predict(x_train_full_2)
print("\nAccuracy for MLP on Training Set:", (r2_score(y_pred_mlp_train, y_train_full)) * 100, "%")
print("Accuracy for MLP on 20% Testing Set:", (r2_score(y_pred_mlp, y_test)) * 100, "%")
end_MLP = time.time()
print("Time Taken: ", end_MLP - start_MLP, "s")