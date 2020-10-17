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


#%%
values = pd.read_csv('AHS2017ValueLabels.csv')
values = values.loc[values['FLAT']=='YES']

print(len(np.unique(values['NAME'])))
print(df.columns)

#%% Variable Lists

# Topic: Admin
OMB13CBSA = '41860'
vars_admin = ['INTSTATUS','SPLITSAMP',]

# Topic: Occupancy and Tenure
vars_occ = ['TENURE','CONDO','HOA','OWNLOT','MGRONSITE','VACRESDAYS','VACRNTDAYS']

# Topic: Structural
vars_struct = ['BLD','NUNITS','YRBUILT','GUTREHB','LOTSIZE','GARAGE','WINBARS','MHWIDE','UNITSIZE','TOTROOMS','KITEXCLU','BATHEXCLU']

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
    

# bath = df['BATHROOMS']
# jbath = df['JBATHROOMS']

# names = df.columns
# print(np.where(names=="DIVISION"))

#%%

from collections import Counter 
n = Counter(df['DPEVLOC'])

# Number of valid features
s = sum([n["'{}'".format(i)] for i in range(1,6)])

# Filter by valid only
df = df.loc[df['DPEVLOC'].isin(["'{}'".format(i) for i in range(1,6)])]
    
#%% Split into train/dev/test set
from sklearn.model_selection import train_test_split

x_vars = [var for var_list in [vars_admin, vars_occ, vars_struct, vars_equip, vars_probs, vars_demo, 
          vars_income, vars_costs, vars_mort, vars_improv, vars_neigh, vars_move, 
          vars_del, vars_dis, vars_comm, vars_evict] for var in var_list]

X = df[df.columns.intersection(x_vars)]
y = df['DPEVLOC']

# Train-val-test = 0.6-0.2-0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#%%
from sklearn.ensemble import RandomForestClassifier

# Have to encode all categorical variables with .astype('category')

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

