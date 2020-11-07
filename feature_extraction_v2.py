#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:42:08 2020

@author: tamrazovd
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

# Load the dataset
df = pd.read_csv('SF_41860_Flat.csv', index_col=0)


#%% Variable Lists

# Data/Variable Types
# Categorial == 1
# Continuous == 0

# Topic: Admin -- a
OMB13CBSA = '41860'
vars_admin = ['INTSTATUS','SPLITSAMP']
type_admin = [1, 1]

# Topic: Occupancy and Tenure
vars_occ = ['TENURE','CONDO','HOA','OWNLOT','MGRONSITE','VACRESDAYS','VACRNTDAYS']
type_occ = [1, 1, 1, 1, 1, 1, 1]

# Topic: Structural
vars_struct = ['BLD','YRBUILT','GUTREHB','GARAGE','WINBARS','MHWIDE','UNITSIZE','TOTROOMS','KITEXCLU','BATHEXCLU']
type_struct = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1]

# Topic: Equipment and Appliances
vars_equip = []

# Topic: Housing Problems
vars_probs = []

# Topic: Demographics
vars_demo = ['HSHLDTYPE','SAMEHHLD','NUMPEOPLE','NUMADULTS','NUMELDERS','NUMYNGKIDS','NUMOLDKIDS','NUMVETS','MILHH','NUMNONREL','PARTNER','MULTIGEN','GRANDHH','NUMSUBFAM','NUMSECFAM','DISHH','HHSEX','HHAGE','HHMAR','HHRACE','HHRACEAS','HHRACEPI','HHSPAN','HHCITSHP','HHNATVTY','HHINUSYR','HHMOVE','HHGRAD','HHENROLL','HHYNGKIDS','HHOLDKIDS','HHADLTKIDS','HHHEAR','HHSEE','HHMEMRY','HHWALK','HHCARE','HHERRND']
type_demo = [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Topic: Income
vars_income = ['HINCP','FINCP','FS']
type_income = [0, 0, 1]

# Topic: Housing Costs
vars_costs = ['MORTAMT','RENT','UTILAMT','PROTAXAMT','INSURAMT','HOAAMT','LOTAMT','TOTHCAMT','HUDSUB','RENTCNTRL','FIRSTHOME','MARKETVAL','TOTBALAMT']
type_costs = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

# Topic: Mortgage Details
vars_mort = []

# Topic: Home Improvement
vars_improv = []

# Topic: Neighborhood Features
vars_neigh = ['SUBDIV','NEARBARCL','NEARABAND','NEARTRASH','RATINGHS','RATINGNH','NHQSCHOOL','NHQPCRIME','NHQSCRIME','NHQPUBTRN','NHQRISK']
type_neigh = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]

# Topic: Recent Movers
vars_move = ['MOVFORCE','MOVWHY','RMJOB','RMOWNHH','RMFAMILY','RMCHANGE','RMCOMMUTE','RMHOME','RMCOSTS','RMHOOD','RMOTHER']
type_move = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Topic: Delinquency
vars_del = []

# Topic: Disaster Planning
vars_dis = ['DPGENERT','DPSHELTR','DPDRFOOD','DPEMWATER','DPEVSEP','DPEVLOC','DPALTCOM','DPGETINFO','DPEVVEHIC','DPEVKIT','DPEVINFO','DPEVFIN','DPEVACPETS','DPFLDINS','DPMAJDIS']
type_dis = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Topic: Commuting
vars_comm = []

# Topic: Eviction
vars_evict = []

# Combine variables in a list
x_vars = np.asarray([var for var_list in [vars_admin, vars_occ, vars_struct, vars_equip,
                               vars_probs, vars_demo, vars_income, vars_costs,
                               vars_mort, vars_improv, vars_neigh, vars_move,
                               vars_del, vars_dis, vars_comm, vars_evict]
          for var in var_list])

x_vars_encode = np.asarray([code for code_list in [type_admin, type_occ, type_struct,
                               type_demo, type_income, type_costs,
                               type_neigh, type_move,
                               type_dis]
          for code in code_list])

#%% Data Cleaning and Filtering

from collections import Counter
n = Counter(df['DPEVLOC'])

# Number of valid features
s = sum([n[f"'{i}'"] for i in range(1,6)])

# M or -9: Not reported
# N or -6: Not applicable

# Filter by valid output only (rows)
df = df.loc[df['DPEVLOC'].isin(["'{}'".format(i) for i in range(1,6)])]

# **NOT REQUIRED IF ENCODING OF MISSING VALUES IS USED**
# Transform MARKETVAL by making all -6 and -9 values = 0
# df['MARKETVAL'] = df['MARKETVAL'].clip(lower=0)

# **NOT REQUIRED IF ENCODING OF MISSING VALUES IS USED**
# Make HHINUSYR a categorical variable
# df['HHINUSYR'] = np.digitize(df['HHINUSYR'], bins=np.arange(-10,2030,10))

# Filter by proportion of NA values (cols)
props_NA = [sum(list(df[var]=="'-6'") or list(df[var]=="'-9'"))/len(df[var]) for var in x_vars]
vars_remove = [x_vars[i] for i, var in enumerate(props_NA) if var > 0.25]

# Exclude certain variables by choice
vars_remove.extend(['MORTAMT','RENT','PROTAXAMT','HOAAMT','LOTAMT','TOTBALAMT'])

# Remove disaster preparedeness variables
vars_remove.extend(vars_dis)

# Remove variables that are constant for all observations (i.e. only 1 unique value)
vars_remove.extend([var for i, var in enumerate(x_vars) if len(np.unique(df[var])) == 1])

# Create a binary list of valid id's and a list of valid variables
idx = [var not in vars_remove for var in x_vars]
valid_vars = x_vars[idx]

# Filter inputs (X), outputs (y), and input variable encoding (X_encode)
X = df[valid_vars]
X_encode = x_vars_encode[idx]
y = df['DPEVLOC']


#%% Encode input and output variables as categorical
from sklearn import preprocessing, metrics, model_selection

# Copy the input array to prevent overwriting
X = X.copy()

# Transform output variable with the LabelEncoder
le = preprocessing.LabelEncoder()
le.fit(np.unique(y))
y = le.transform(y)

# Loop through each input variable and encode categorical ones (i.e. X_incode == 1)
for i, val in enumerate(X_encode):
    col = valid_vars[i]
    Xi = X.loc[:,col]
    
    if val == 1:
        # Option #1: encode categorical variables as One Hot encoder
        OneHot = pd.get_dummies(Xi, prefix=col)
        if OneHot.shape[1] <= 20:
            X = pd.concat([X, OneHot], axis=1)
        X = X.drop(col, axis=1)
        
        # Option #2: encode categorical variables as Label encoder
        # Xi = X.loc[:,col]
        # if len(np.unique(Xi)) <= 5:
        #     le.fit(np.unique(Xi))
        #     X.loc[:,col] = le.transform(Xi)
        # else: X = X.drop(col, axis=1)
        
    # **Optional** 
    # Encoding of missing values in non-categorical variables
    # If the a missing value is present in the variable (i.e. -6 or -9), 
    # a separate index variable is created to represent missing values, while
    # -6 and -9 are replaced with 0 in the continuous variable.

    elif val == 0:
        if any(Xi < 0):
            le.fit([0,1])
            X[col + '_MISSING'] = le.transform([i < 0 for i in X[col]])
            X[col] = X[col].clip(lower=0)
            if col in valid_vars: np.append(valid_vars, [col + '_MISSING'])

#%% Split into train/dev/test set
# Train-val-test ratio = 0.6-0.2-0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#%% Ridge regression classifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix

clf = RidgeClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

print(clf.score(X_val, y_val))
print(balanced_accuracy_score(y_val, clf.predict(X_val)))
print(confusion_matrix(y_val , clf.predict(X_val)))

#%% Synthesize additional observations for all but majority class
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='not majority')
X_train, y_train = smote.fit_sample(X_train, y_train)

#%%
# np.save('X_train', X_train)
# np.save('X_val', X_val)
# np.save('X_test', X_test)
# np.save('y_train', y_train)
# np.save('y_val', y_val)
# np.save('y_test', y_test)

# # Pandas dataframe for linear regression (using label-encoded inputs)
# pd_xtrain = pd.DataFrame(data = X_train, columns = X.columns)
# pd_xval = pd.DataFrame(data = X_val, columns = X.columns)
# pd_xtest = pd.DataFrame(data = X_test, columns = X.columns)
# pd_ytrain = pd.DataFrame(data = y_train, columns = ['Target'])
# pd_yval = pd.DataFrame(data = y_val, columns = ['Target'])
# pd_ytest = pd.DataFrame(data = y_test, columns = ['Target'])

# pd_xtrain.to_pickle('pd_X_train')
# pd_xval.to_pickle('pd_X_val')
# pd_xtest.to_pickle('pd_X_test')
# pd_ytrain.to_pickle('pd_y_train')
# pd_yval.to_pickle('pd_y_val')
# pd_ytest.to_pickle('pd_y_test')

#%%
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=500, 
                      eta=0.05, 
                      max_depth=3, 
                      colsample_bytree=0.3,
                      reg_lambda=1e4,
                      subsample=0.6)
model.fit(X_train, y_train)

#%%
y_pred = model.predict(X_val)

print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))