#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:42:08 2020

@author: tamrazovd
"""

import pandas as pd 
import numpy as np
import statsmodels.formula.api as sm

df = pd.read_csv('SF_41860_Flat.csv', index_col=0)


#%% Variable Lists

# Topic: Admin: 


#%%

from collections import Counter 
n = Counter(df['DPEVLOC'])

# Number of valid features
s = sum([n["'{}'".format(i)] for i in range(1,6)])
    
#%% Split into train/dev/test set
from sklearn.model_selection import train_test_split

x_vars = ['TOTROOM', 'PERPOVLVL', 'COMTYPE']
X = df[df.columns.intersection(x_vars)]
y = df['DPEVLOC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
