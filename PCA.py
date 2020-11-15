#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:08:16 2020

@author: Aaron
"""


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('SF_41860_Flat.csv', index_col=0)
# df2 = pd.read_csv('SJ_41940_Flat.csv', index_col=0)
# df3 = pd.read_csv('CA_41860_31080_Flat.csv', index_col=0)

# df = pd.concat([df1,df2], ignore_index=True)
# df = pd.concat([df2,df3], ignore_index=True)
df = df1


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

n = Counter(df['DPEVLOC'])

# Number of valid features
# n = np.asarray([n[f"'{i}'"] for i in range(1,6)])
n = np.asarray([n[f"'{i}'"] for i in range(1,4)])
w = sum(n)/n

# M or -9: Not reported
# N or -6: Not applicable

# Filter by valid output only (rows)
# df = df.loc[df['DPEVLOC'].isin(["'{}'".format(i) for i in range(1,6)])]
df = df.loc[df['DPEVLOC'].isin(["'{}'".format(i) for i in range(1,4)])]

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

# y = pd.get_dummies(y)

# Loop through each input variable and encode categorical ones (i.e. X_incode == 1)
for i, val in enumerate(X_encode):
    col = valid_vars[i]
    Xi = X.loc[:,col]

    if val == 1:
        # Option #1: encode categorical variables as One Hot encoder
        # OneHot = pd.get_dummies(Xi, prefix=col)
        # if OneHot.shape[1] <= 20:
        #     X = pd.concat([X, OneHot], axis=1)
        #     X_encode = np.append(X_encode, np.repeat(val, OneHot.shape[1]))
        # X = X.drop(col, axis=1)
        # X_encode = np.delete(X_encode, i)

                
        #Option #2: encode categorical variables as Label encoder
        Xi = X.loc[:,col]
        le.fit(np.unique(Xi))
        X.loc[:,col] = le.transform(Xi)
        X[col] = X[col].astype('category')
        # X = X.drop(col, axis=1)
        
    # *Optional* 
    # Encoding of missing values in non-categorical variables
    # If the a missing value is present in the variable (i.e. -6 or -9), 
    # a separate index variable is created to represent missing values, while
    # -6 and -9 are replaced with 0 in the continuous variable.

    elif val == 0:
        if any(Xi < 0):
            le.fit([0,1])
            X[col + '_MISSING'] = le.transform([i < 0 for i in X[col]])
            X[col] = X[col].clip(lower=0)
            valid_vars = np.append(valid_vars, [col + '_MISSING'])
            X_encode = np.append(X_encode, val) 
            
#%% PCA

n_components=2

# Standardizing the features
x = StandardScaler().fit_transform(X)
# PCA projection to 2D
pca = PCA(n_components) 
pccols = ['PC1', 'PC2']
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = pccols)
# finalDf = pd.concat([principalDf, y], axis = 1)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.get_params())


# Plot transformed data
a = plt.figure(dpi=300)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.title('2 Component PCA', fontsize = 16)
plt.scatter(principalDf['PC1'], principalDf['PC2'], c = y,cmap='viridis', s = 4)
plt.legend()
# plt.colorbar()#.set_label(p_names[target])
plt.show()
a.savefig('PCA.png',dpi=300)
plt.close()

# Get component loadings 
# (correlation coefficient between original variables and the component) 
loadings = pca.components_
loadings_df = pd.DataFrame.from_dict(dict(zip(pccols, loadings)))
loadings_df['Input'] = X.columns
loadings_df = loadings_df.set_index('Input')
loadings_df['Norm'] = np.sqrt(np.square(loadings_df.iloc[:,:2]).sum(axis=1))
loadings_df['Abs(PC1)'] = np.abs(loadings_df['PC1'])
loadings_df['Abs(PC2)'] = np.abs(loadings_df['PC2'])
loadings_df['ScaledVar'] = \
    (np.square(loadings_df.iloc[:,:2]) * pca.explained_variance_ratio_).sum(axis=1)

# Rank by norm of coefficients for PC1 and PC2:
rank_norm = loadings_df.sort_values(by=['Norm'],ascending=False)
top10_norm = rank_norm[:10].index
# Rank by PC1 most important: max abs value
rank_PC1 = loadings_df.sort_values(by=['Abs(PC1)'],ascending=False)
top5_PC1 = rank_PC1[:5]
# Rank by PC2 most important: max abs value
rank_PC2 = loadings_df.sort_values(by=['Abs(PC2)'],ascending=False)
top5_PC2 = rank_PC2[:5]
# Rank by squared coefficients scaled by explained variance ratio
rank_scaledvar = loadings_df.sort_values(by=['ScaledVar'],ascending=False)
top5_scaledvar = rank_scaledvar[:10]


# THIS GRAPH IS NOT VERY USEFUL WITH SO MANY VARIABLES:
# # Get correlation matrix plot for loadings
# a = plt.figure()
# ax = sns.heatmap(loadings_df, annot=True, cmap='PRGn')
# sns.set(font_scale=1.5)
# ax.set_ylabel('')
# plt.tight_layout()
# plt.show()
# a.savefig('PlantsCV.png',dpi=300,width=6,height=4)
# plt.close()