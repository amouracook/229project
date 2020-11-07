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

selector = RFE(estimator, n_features_to_select=30, step=1, verbose=0)
selector = selector.fit(X_train, y_train)
selected_features = np.take(features, np.where(selector.support_)[0])
print(selected_features)
print(selector.score(X_train,y_train))
print(selector.score(X_val,y_val))


X = sm.add_constant(X_train[selected_features])
model = sm.OLS(y_train,X)
results = model.fit()
print(results.summary())
qq = sm.qqplot(results.resid,line="s",markersize=3)
plt.tight_layout()
# # qq.savefig('QQ.png',dpi=300)