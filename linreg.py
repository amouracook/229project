#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:31:01 2020

@author: Aaron
"""
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.feature_selection import RFE, RFECV # recursive feature selection
from sklearn.linear_model import LinearRegression # estimator
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# X_train = np.load('Data Label Encoded/X_train.npy')
# X_val = np.load('Data Label Encoded/X_val.npy')
# X_test = np.load('Data Label Encoded/X_test.npy')
# y_train = np.load('Data Label Encoded/y_train.npy')
# y_val = np.load('Data Label Encoded/y_val.npy')
# y_test = np.load('Data Label Encoded/y_test.npy')


X_train = pd.read_pickle('Pandas Dataframes/pd_X_train')
X_val = pd.read_pickle('Pandas Dataframes/pd_X_val')
X_test = pd.read_pickle('Pandas Dataframes/pd_X_test')
y_train = pd.read_pickle('Pandas Dataframes/pd_y_train')
y_val = pd.read_pickle('Pandas Dataframes/pd_y_val')
y_test = pd.read_pickle('Pandas Dataframes/pd_y_test')

features = X_train.columns

# Algin indices
y_train = y_train.set_index(X_train.index)

estimator = LinearRegression()

selector = RFE(estimator, n_features_to_select=35, step=1, verbose=0)
selector = selector.fit(X_train, y_train)
selected_features = np.take(features, np.where(selector.support_)[0])
print(selected_features)
print(selector.score(X_train,y_train))
print(selector.score(X_val,y_val))

# Using statsmodels
# X = sm.add_constant(X_train[selected_features])
X = X_train[selected_features]
model = sm.OLS(y_train,X)
results = model.fit()
print(results.summary())
qq = sm.qqplot(results.resid,line="s",markersize=3)
plt.tight_layout()


ypred = model.predict(X) # something wrong
print(accuracy_score(y_val, ypred))
print(confusion_matrix(y_val, ypred))

#%%
# Using sklearn
clf = LinearRegression()
clf.fit(X, y_train)
print(clf.score(X_val[selected_features], y_val))
print(accuracy_score(y_val, clf.predict(X_val[selected_features])))
print(confusion_matrix(y_val , clf.predict(X_val[selected_features])))

#%%

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=3)
Xp = polynomial_features.fit_transform(X)
model = sm.OLS(y_train,Xp)
results = model.fit()
print(results.summary())
