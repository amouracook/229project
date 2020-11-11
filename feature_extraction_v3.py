#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 09:42:08 2020

@author: tamrazovd
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
# from imblearn.over_sampling import SMOTE
from imblearn import over_sampling as os
from sklearn import preprocessing, metrics, model_selection
from collections import Counter

# Load the dataset
df = pd.read_csv('SF_41860_Flat.csv', index_col=0)
# df = pd.read_csv('CA_41860_31080_Flat.csv', index_col=0)


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
# s = sum([n[f"'{i}'"] for i in range(1,4)])

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

#%% Split into train/dev/test set
# Train-val-test ratio = 0.6-0.2-0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=0)

smote = os.SMOTENC(categorical_features = X_encode.astype('bool'),  
                    sampling_strategy='not majority',
                    random_state=0)
# smote = os.SMOTE(sampling_strategy='not majority')

X_train, y_train = smote.fit_sample(X_train, y_train)

X = pd.concat([X_train, X_val, X_test], ignore_index= True)
y = pd.concat([y_train, y_val, y_test], ignore_index= True)

train_sep = X_train.shape[0]
val_sep = train_sep + X_val.shape[0]
test_sep = val_sep + X_test.shape[0]

# Data augmentation using SMOTE
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
# X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=1)

# smote = SMOTE(sampling_strategy='not majority',random_state=1)
# X_train, y_train = smote.fit_sample(X_train, y_train)

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
            
X_train, y_train = X[0:train_sep], y[0:train_sep]
X_val, y_val = X[train_sep:val_sep], y[train_sep:val_sep]
X_test, y_test = X[val_sep:test_sep], y[val_sep:test_sep]  
            
#%% Make all variables categorical
# Code copied from: https://jovian.ai/aakanksha-ns/shelter-outcome

for col in X.loc[:,X_encode==1].columns:
    X[col] = X[col].astype('category')

#%% Categorical embedding for categorical columns having more than two values

# Choosing columns for embedding
embedded_cols = {n: len(col.cat.categories) for n,col in X.loc[:,X_encode==1].items() if len(col.cat.categories) > 2}
embedded_col_names = embedded_cols.keys()

# Determinining size of embedding
# (borrowed from https://www.usfca.edu/data-institute/certificates/fundamentals-deep-learning lesson 2)
embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]

# np.save('X_train', X_train)
# np.save('X_val', X_val)
# np.save('X_test', X_test)
# np.save('y_train', y_train)
# np.save('y_val', y_val)
# np.save('y_test', y_test)

#%% Implement neural net
# Code copied from: https://jovian.ai/aakanksha-ns/shelter-outcome

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DisasterPreparednessDataset(Dataset):
    def __init__(self, X, Y, embedded_col_names):
        X = X.copy()
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32) #numerical columns
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

# Creating train and valid datasets
train_ds = DisasterPreparednessDataset(X_train, y_train, embedded_col_names)
valid_ds = DisasterPreparednessDataset(X_val, y_val, embedded_col_names)

#%% Making device GPU/CPU compatible
# (borrowed from https://jovian.ml/aakashns/04-feedforward-nn)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
device = get_default_device()

#%% Model
# From: https://www.usfca.edu/data-institute/certificates/fundamentals-deep-learning lesson 2

class DisasterPreparednessModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) 
                                         for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        D1 = self.n_emb + self.n_cont
        D2 = 2*(self.n_emb + self.n_cont)//3 + 3
        # D3 = 4*(self.n_emb + self.n_cont)//12
        D4 = 3
        self.lin1 = nn.Linear(D1, D2) #just CS things
        # self.lin2 = nn.Linear(D2, D3)
        self.lin3 = nn.Linear(D2, D4)
        self.bn1 = nn.BatchNorm1d(self.n_cont) #n_cont = number of cont. features
        # self.bn2 = nn.BatchNorm1d(D2)
        self.bn3 = nn.BatchNorm1d(D2)
        self.emb_drop = nn.Dropout(0.2) # experiment with dropout probability
        self.drops = nn.Dropout(0.5)
        
        # self.emb_drop = nn.Dropout(0.6) # experiment with dropout probability
        # self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        # self.lin2 = nn.Linear(200, 70) # play around with hidden layer sizes
        # self.lin3 = nn.Linear(70, 5)
        # self.drops = nn.Dropout(0.3)
        # self.softmax = nn.Softmax(dim=1) # we added this
        

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        # x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        # x = self.drops(x)
        # x = self.bn2(x)
        # x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        # x = self.softmax(x) # we added this
        return x

#%% More function definition

# Optimizer
def get_optimizer(model, lr = 0.001, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim

# Training function
def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0] # size of batch
        output = model(x1, x2) # forward pass
        loss = F.cross_entropy(output, y)   
        optim.zero_grad() #don't accumulate gradients in the optimizer object
        loss.backward() # calculate gradient (backward pass)
        optim.step() # take gradient descent step
        total += batch # add batch loss to total loss
        sum_loss += batch*(loss.item())
    return sum_loss/total

# Evaluation function
def val_loss(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    pred_out = []
    y_out = []
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = F.cross_entropy(out, y)
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        pred_out = np.hstack((pred_out,np.asarray(pred)))
        y_out = np.hstack((y_out,np.asarray(y)))
        correct += (pred == y).float().sum().item()
    print("valid loss %.3f, total accuracy %.3f, and balanced accuracy %.3f" % (sum_loss/total, correct/total, balanced_accuracy_score(y_out, pred_out)))
    
    return sum_loss/total, correct/total

def train_loop(model, epochs, lr=0.01, wd=0.0):
    optim = get_optimizer(model, lr = lr, wd = wd)
    for i in range(epochs): 
        loss = train_model(model, optim, train_dl)
        # print("training loss: ", loss)
        val_loss(model, valid_dl)
        
#%% Training #%% Model & training set-up
model = DisasterPreparednessModel(embedding_sizes, X.shape[1]-len(embedded_cols))
to_device(model, device)

# Do we want to batch it?
batch_size = 50
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)

#%%
# train_dl = DeviceDataLoader(train_dl, device)
# valid_dl = DeviceDataLoader(valid_dl, device)
train_loop(model, epochs=200, lr=1e-5, wd=1e-1)

#%% Test output
model.eval()
test_ds = DisasterPreparednessDataset(X_val, y_val, embedded_col_names)
test_dl = DataLoader(test_ds, batch_size=batch_size)

preds = []
with torch.no_grad():
    for x1,x2,y in test_dl:
        out = model(x1, x2)
        prob = F.softmax(out, dim=1)
        preds.append(prob)

y_pred = [torch.argmax(item).item() for sublist in preds for item in sublist]     
print(balanced_accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
