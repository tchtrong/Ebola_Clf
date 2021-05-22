#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:45:04 2021

@author: binn

"""

#%%
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd

#%%
#Label for all cases
label = pd.read_csv("../csv/labels.csv")
y_train, y_test = train_test_split(label["0"], test_size=0.2, random_state=123)

#%%
#SVM only

#Read dataset
dataset = pd.read_csv("../csv/freq_matrix_without_X.csv")

#%%
#Split dataset into training and testing set
X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=123)

#%%
#Use SVM
clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

#%%
#LDA 6tp + SVM

#Read dataset
dataset_6tp = pd.read_csv("csv/lda_6tp_without_X.csv", index_col=(0))

#Split dataset into training and testing set
X_6tp_train, X_6tp_test = train_test_split(dataset_6tp, test_size=0.2, random_state=123)

#Use SVM
clf = svm.SVC()
clf.fit(X_6tp_train, y_train)
print(clf.score(X_6tp_test, y_test))

#%%
for i in range(10, 310, 10):
    dataset = pd.read_csv("csv/lda_{}tp_without_X.csv".format(i), index_col=(0))
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=123)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))



