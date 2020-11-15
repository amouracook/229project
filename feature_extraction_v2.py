#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:42:08 2020

@author: tamrazovd
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from imblearn import over_sampling as os
from collections import Counter
from sklearn import preprocessing, metrics, model_selection
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix
# from xgboost import XGBClassifier
import matplotlib.pylab as plt
from matplotlib import pyplot
# from xgboost import plot_importance, plot_tree



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

# Count the number of responses of each kind to DPEVLOC (1-5, -6 or -9)
# M or -9: Not reported
# N or -6: Not applicable
n = Counter(df['DPEVLOC'])

# Number of valid features
# s = sum([n[f"'{i}'"] for i in range(1,4)])

# Filter by valid output only (rows) -- keep just responses 1, 2, and 3
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


#%% Split into train/dev/test set
# Train-val-test ratio = 0.6-0.2-0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=0)

# Use SMOTE for continuous features to oversample the non-majority classes
smote = os.SMOTENC(categorical_features = X_encode.astype('bool'),  
                    sampling_strategy='not majority',
                    random_state=0)
# smote = os.SMOTE(sampling_strategy='not majority')

X_train, y_train = smote.fit_sample(X_train, y_train)

# Concatenate all three sets into master X and y dataframes
X = pd.concat([X_train, X_val, X_test], ignore_index= True)
y = pd.concat([y_train, y_val, y_test], ignore_index= True)

# Indices to separate out the three sets
train_sep = X_train.shape[0]
val_sep = train_sep + X_val.shape[0]
test_sep = val_sep + X_test.shape[0]

#%% Encode input and output variables as categorical

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
        # X = X.drop(col, axis=1)
        
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
            valid_vars = np.append(valid_vars, [col + '_MISSING'])
            X_encode = np.append(X_encode, val)

# After encoding, split back into train, val, and test sets
X_train, y_train = X[0:train_sep], y[0:train_sep]
X_val, y_val = X[train_sep:val_sep], y[train_sep:val_sep]
X_test, y_test = X[val_sep:test_sep], y[val_sep:test_sep]    

    
#%% Ridge regression classifier
''' NOTE: works better with label encoding instead of one hot '''

clf = RidgeClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

print(clf.score(X_val, y_val))
print(balanced_accuracy_score(y_val, clf.predict(X_val)))
print(confusion_matrix(y_val , clf.predict(X_val)))

#%% Save the data frames
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

def f1_eval(y_pred, dtrain):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, y_pred, average='weighted')
    # err = balanced_accuracy_score(y_true,y_pred)
    return 'f1_err', err

def balanced_score(y_pred, dtrain):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = dtrain.get_label()
    err = balanced_accuracy_score(y_true,y_pred)
    return 'balanced_accuracy', err
# Just SF
# model = XGBClassifier(n_estimators=100, 
#                       eta=0.1, 
#                       max_depth=3, 
#                       colsample_bytree=0.3,
#                       reg_lambda=1e1,
#                       subsample=0.2)

# model = XGBClassifier(n_estimators=250, 
#                       eta=0.01, 
#                       max_depth=2,
#                       colsample_bytree=0.2,
#                       reg_lambda=1e2,
#                       subsample=0.1,
#                       random_state=0,
#                       objective='multi:softmax')

# model = XGBClassifier(booster='dart',
#                       n_estimators=500, 
#                       eta=0.01, 
#                       max_depth=5,
#                       colsample_bytree=0.2,
#                       alpha=10,
#                       # reg_lambda=1e2,
#                       subsample=0.2,
#                       random_state=0,
#                       objective='multi:softmax',
#                       rate_drop=0.5,
#                       skip_drop=0.25)

model = XGBClassifier(n_estimators=500, 
                      eta=0.1, 
                      max_depth=3,
                      colsample_bytree=0.1,
                      reg_lambda=5e2,
                      subsample=0.3,
                      random_state=4)

model.fit(X_train, y_train, eval_metric=['merror', 'mlogloss'],
          eval_set=[(X_train, y_train), (X_val, y_val)], 
          # early_stopping_rounds=50, 
          verbose=False)

y_pred = model.predict(X_val)

print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

#%%
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#%% Importance graph
fig, ax = plt.subplots(1, 1, dpi=300)
fig.set_size_inches(8,5)

ax.minorticks_on()

plot_importance(model, max_num_features=15, color='k', grid=False, ax=ax, zorder=10) # top 10 most important features

ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
ax.grid(b=True, which='minor', axis='x', color='gray', linestyle='-', linewidth=0.25, alpha=0.25, zorder=0)


fig.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300)

plt.show()

#%%
plot_tree(model, num_trees=0, dpi=300)
plt.show()

#%% Convergence plot

# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

# plot log loss
# fig, ax = plt.subplots(1, 1, dpi=300)
# fig.set_size_inches(8,5)

# ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
# ax.legend()
# pyplot.ylabel('Log Loss')
# pyplot.title('XGBoost Log Loss')
# pyplot.show()
# plot classification error
fig, ax = plt.subplots(1, 1, dpi=300)
fig.set_size_inches(8,5)
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Validation')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

#%%
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=3, 
                               random_state=2,
                               n_estimators=100,
                               min_samples_leaf=3,
                               max_features='sqrt')

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

#%%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0, 
                           multi_class='multinomial', 
                           penalty='l2', max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))


