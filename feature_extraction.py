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

x_vars = [var for var_list in [vars_admin, vars_occ, vars_struct, vars_equip,
                               vars_probs, vars_demo, vars_income, vars_costs,
                               vars_mort, vars_improv, vars_neigh, vars_move,
                               vars_del, vars_dis, vars_comm, vars_evict]
          for var in var_list]

encode = [code for code_list in [type_admin, type_occ, type_struct,
                               type_demo, type_income, type_costs,
                               type_neigh, type_move,
                               type_dis]
          for code in code_list]

#%% Data Cleaning and Filtering

from collections import Counter
n = Counter(df['DPEVLOC'])

# Number of valid features
s = sum([n[f"'{i}'"] for i in range(1,6)])

# M or -9: Not reported
# N or -6: Not applicable

# Filter by valid only (rows)
df = df.loc[df['DPEVLOC'].isin(["'{}'".format(i) for i in range(1,6)])]

# Filter by proportion of NA values (cols)
props_NA = [sum(list(df[var]=="'-6'") or list(df[var]=="'-9'"))/len(df[var]) for var in x_vars]
ind_remove = [i for i, var in enumerate(props_NA) if var > 0.25]

# Transform MARKETVAL by making all -6 and -9 values = 0
df = df['MARKETVAL'].replace(-6, 0)
df = df['MARKETVAL'].replace(-9, 0)

# Make HHINUSYR a categorical variable
df['HHINUSYR'] = np.digitize(df['HHINUSYR'], bins=np.arange(-10,2030,10))

# Exclude certain variables by choice
vars_remove = ['MORTAMT','RENT','PROTAXAMT','HOAAMT','LOTAMT','TOTBALAMT']
vars_remove.extend(vars_dis)
ind_remove.extend([i for i, var in enumerate(x_vars) if var in vars_remove and i not in ind_remove])

# Final list of variables to keep
ind_keep = list(range(len(x_vars)))
[ind_keep.remove(i) for i in ind_remove]
good_vars = [x_vars[i] for i in ind_keep]

#%% Split into train/dev/test set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

X = df[x_vars]
y = df['DPEVLOC']

encode = [code for code_list in [type_admin, type_occ, type_struct,
                               type_demo, type_income, type_costs,
                               type_neigh, type_move] for code in code_list]

X = X.copy()
le = preprocessing.LabelEncoder()
for i, val in enumerate(encode):
    print(x_vars[i],val)
    if val == 1:
        col = x_vars[i]
        Xi = X.loc[:,col]
        le.fit(np.unique(Xi))
        X.loc[:,col] = le.transform(Xi)

le.fit(np.unique(y))
y = le.transform(y)
# Filter by only good variables
X = X.loc[:,good_vars]

#%%
# Train-val-test = 0.6-0.2-0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#%%
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_train, y_train)
accuracy = sum(y_val == clf.predict(X_val))/y_val.shape[0]
print(accuracy)
