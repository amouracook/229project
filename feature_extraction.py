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


#%% Data Categorization

values = pd.read_csv('AHS2017ValueLabels.csv')
values = values.loc[values['FLAT']=='YES']

print(len(np.unique(values['NAME'])))
print(df.columns)

#%% Variable Lists

# Data/Variable Types
# Categorial == 1
# Continuous == 0

# Topic: Admin
OMB13CBSA = '41860'
vars_admin = ['INTSTATUS','SPLITSAMP']
type_admin = [1, 1]

# Topic: Occupancy and Tenure
vars_occ = ['TENURE','CONDO','HOA','OWNLOT','MGRONSITE','VACRESDAYS','VACRNTDAYS']
type_occ = [1, 1, 1, 1, 1, 1, 1]

# Topic: Structural
vars_struct = ['BLD','YRBUILT','GUTREHB','LOTSIZE','GARAGE','WINBARS','MHWIDE','UNITSIZE','TOTROOMS','KITEXCLU','BATHEXCLU']
type_struct = [1, 0, 1, ]

# Topic: Equipment and Appliances
vars_equip = []

# Topic: Housing Problems
vars_probs = []

# Topic: Demographics
vars_demo = ['HSHLDTYPE','SAMEHHLD','NUMPEOPLE','NUMADULTS','NUMELDERS','NUMYNGKIDS','NUMOLDKIDS','NUMVETS','MILHH','NUMNONREL','SAMSEXHH','PARTNER','MULTIGEN','GRANDHH','NUMSUBFAM','NUMSECFAM','DISHH','HHSEX','HHAGE','HHMAR','HHRACE','HHRACEAS','HHRACEPI','HHSPAN','HHCITSHP','HHNATVTY','HHINUSYR','HHMOVE','HHGRAD','HHENROLL','HHYNGKIDS','HHOLDKIDS','HHADLTKIDS','HHHEAR','HHSEE','HHMEMRY','HHWALK','HHCARE','HHERRND']

# Topic: Income
vars_income = ['HINCP','FINCP','FS']

# Topic: Housing Costs
vars_costs = ['MORTAMT','RENT','UTILAMT','PROTAXAMT','INSURAMT','HOAAMT','LOTAMT','TOTHCAMT','HUDSUB','RENTCNTRL','FIRSTHOME','MARKETVAL','TOTBALAMT']

# Topic: Mortgage Details
vars_mort = []

# Topic: Home Improvement
vars_improv = []

# Topic: Neighborhood Features
vars_neigh = ['SUBDIV','NEARBARCL','NEARABAND','NEARTRASH','RATINGHS','RATINGNH','NHQSCHOOL','NHQPCRIME','NHQSCRIME','NHQPUBTRN','NHQRISK']

# Topic: Recent Movers
vars_move = ['MOVFORCE','MOVWHY','RMJOB','RMOWNHH','RMFAMILY','RMCHANGE','RMCOMMUTE','RMHOME','RMCOSTS','RMHOOD','RMOTHER']

# Topic: Delinquency
vars_del = []

# Topic: Disaster Planning
vars_dis = ['DPGENERT','DPSHELTR','DPDRFOOD','DPEMWATER','DPEVSEP','DPEVLOC','DPALTCOM','DPGETINFO','DPEVVEHIC','DPEVKIT','DPEVINFO','DPEVFIN','DPEVACPETS','DPFLDINS','DPMAJDIS']

# Topic: Commuting
vars_comm = []

# Topic: Eviction
vars_evict = []

x_vars = [var for var_list in [vars_admin, vars_occ, vars_struct, vars_equip,
                               vars_probs, vars_demo, vars_income, vars_costs,
                               vars_mort, vars_improv, vars_neigh, vars_move,
                               vars_del, vars_dis, vars_comm, vars_evict]
          for var in var_list]

#%% Data Cleaning

from collections import Counter 
n = Counter(df['DPEVLOC'])

# Number of valid features
s = sum([n[f"'{i}'"] for i in range(1,6)])

# M or -9: Not reported
# N or -6: Not applicable

# Filter by valid only
df = df.loc[df['DPEVLOC'].isin(["'{}'".format(i) for i in range(1,6)])]

# Get proportion of nonreported values in the features
props_NA = [sum(list(df[var]=="'-6'") or list(df[var]=="'-9'"))/len(df[var]) for var in x_vars]
badvars_i = [i for i, var in enumerate(props_NA) if var > 0.25]
badvars = [x_vars[i] for i in badvars_i]
    
#%% Split into train/dev/test set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

x_vars_update = df.columns.intersection(x_vars)
X = df[x_vars]
y = df['DPEVLOC']

encode = [1 for i in x_vars_update]

le = preprocessing.LabelEncoder()
for i, val in enumerate(encode):
    if val == 1:
        col = x_vars[i]
        Xi = X.loc[:,col].copy()
        le.fit(np.unique(Xi))
        X.loc[:,col] = le.transform(Xi)

#%%
# Train-val-test = 0.6-0.2-0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#%%
from sklearn.ensemble import RandomForestClassifier

# Have to encode all categorical variables with .astype('category')

clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(X_train, y_train)
print(sum(y_val == clf.predict(X_val))/y_val.shape[0])


# Train-val-test = 0.6-0.2-0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#%%
from sklearn.ensemble import RandomForestClassifier

# Have to encode all categorical variables with .astype('category')

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

