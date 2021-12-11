# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import confusion_matrix


#load images
img = load_digits()


#plt.imshow(np.reshape(digits.data[0], (8, 8)), cmap='gray')
#plt.title('Label: %i\n' % digits.target[0], fontsize=25)
#get image's data and test
X = img.data
y = img.target

#test size
ts = 0.28

#split image by using sklearn’s train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=1)

#initialize KNeighborsClassifier
knn = KNeighborsClassifier()
# classfy
knn.fit(X_train, y_train)
# calculate predict
ky_predict = knn.predict(X_test)
# get mean of predict
Kacc = np.mean(y_test == ky_predict)
# get recalla
K_recall = metrics.recall_score(y_test, ky_predict, average='micro')
# get matrix
Kmatric = confusion_matrix(y_test, ky_predict)

#initialize SGDClassifier
sgd = SGDClassifier()
# classfy
sgd.fit(X_train, y_train)
# calculate predict
sy_predict = sgd.predict(X_test)
# get mean of predict
Sacc = np.mean(y_test == sy_predict)
# get recall
S_recall = metrics.recall_score(y_test, sy_predict, average='micro')
# get matrix
Smatric = confusion_matrix(y_test, sy_predict)

#initialize DecisionTreeClassifier
dec = DecisionTreeClassifier()
# classfy
dec.fit(X_train, y_train)
# calculate predict
dy_predict = dec.predict(X_test)
# get mean of predict
Dacc = np.mean(y_test == dy_predict)
# get recall
D_recall = metrics.recall_score(y_test, dy_predict, average='micro')
# get matrix
Dmatric = confusion_matrix(y_test, dy_predict)

print("COMP9517 Week 5 Lab - z5212125\n")
print("Test size = ",ts)
print("KNN Accuracy: ""{:.3f}".format(Kacc),"      Recall: ""{:.3f}".format(K_recall))
print("SGD Accuracy: ""{:.3f}".format(Sacc),"      Recall: ""{:.3f}".format(S_recall))
print("DT Accuracy: ""{:.3f}".format(Dacc),"      Recall: ""{:.3f}".format(D_recall))
print()
max_of_acc = max(Kacc, Sacc, Dacc)
if max_of_acc == Kacc:
    print("KNN confusion Matrics:")
    print(Kmatric)
elif max_of_acc == Sacc:
    print("SGD confusion Matrics:")
    print(Smatric)
else :
    print("DT confusion Matrics:")
    print(Dmatric)





