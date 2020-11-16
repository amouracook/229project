#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:51:34 2020

@author: Davyd, Ana, Aaron
"""

from feature_extraction import feature_extraction
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import kornia as kr
from scipy.optimize import minimize

torch.manual_seed(1)
# np.random.seed(0)

# dataset: 0 = SF data only, 1 = SF + LA data, 2 = SF + SJ data, 3 = All of CA

X, X_encode, X_train, y_train, X_val, y_val, X_test, y_test, n = \
    feature_extraction(dataset = 0, onehot_option = True, smote_option = False, y_stratify=True, seed=0)
    
    
#%% Categorical embedding for categorical columns having more than two values

# Choosing columns for embedding
embedded_cols = {n: len(col.cat.categories) for n,col in X.loc[:,X_encode==1].items() if len(np.unique(col)) > 20}
embedded_col_names = embedded_cols.keys()

# Determinining size of embedding
# (borrowed from https://www.usfca.edu/data-institute/certificates/fundamentals-deep-learning lesson 2)
embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]

#%% Implement neural net
# Code copied from: https://jovian.ai/aakanksha-ns/shelter-outcome

class DisasterPreparednessDataset(Dataset):
    def __init__(self, X, Y, embedded_col_names):
        X = X.copy()
        #categorical columns
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64)
        #numerical columns
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32)
        #target
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
        D2 = 50 #2*(self.n_emb + self.n_cont)//3 + 3
        D3 = 3
        self.lin1 = nn.Linear(D1, D2) #just CS things
        self.lin2 = nn.Linear(D2, D3)
        self.bn1 = nn.BatchNorm1d(self.n_cont) # n_cont = number of cont. features
        self.bn2 = nn.BatchNorm1d(D2)
        self.emb_drop = nn.Dropout(0.2) # dropout probability for features
        self.drops = nn.Dropout(0.5) # dropout probability for hidden layers
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x

#%% More function definition

# Optimizer
def get_optimizer(model, lr = 0.001, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim

# Training function
def train_model(model, optim, train_dl, LL):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0] # size of batch
        output = model(x1, x2) # forward pass
        loss = LL(output,y)
        optim.zero_grad() #don't accumulate gradients in the optimizer object
        loss.backward() # calculate gradient (backward pass)
        optim.step() # take gradient descent step
        total += batch # add batch loss to total loss
        sum_loss += batch*(loss.item())
    return sum_loss/total

# Evaluation function
def val_loss(model, valid_dl, LL):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    pred_out = []
    y_out = []
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = LL(out,y)
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        pred_out = np.hstack((pred_out,np.asarray(pred)))
        y_out = np.hstack((y_out,np.asarray(y)))
        correct += (pred == y).float().sum().item()
    print("valid loss %.3f, total accuracy %.3f, and balanced accuracy %.3f" % 
          (sum_loss/total, correct/total, balanced_accuracy_score(y_out, pred_out)))
    
    return sum_loss/total, correct/total

def train_loop(model, LL, epochs, lr=0.01, wd=0.0):
    optim = get_optimizer(model, lr = lr, wd = wd)
    for i in range(epochs): 
        loss = train_model(model, optim, train_dl, LL)
        print("training loss: ", loss)
        val_loss(model, valid_dl, LL)

def predict(outs,w):
    if type(w).__module__ == np.__name__:
        # if w is a numpy array convert to tensor
        w = torch.from_numpy(w)
    weights = [torch.mul(outs[batch],w) for batch in range(len(outs))]
    preds = [torch.argmax(item).item() for sublist in weights for item in sublist]
    return preds
        
#%% Model 

batch_size = 100

model = DisasterPreparednessModel(embedding_sizes, X.shape[1]-len(embedded_cols))
to_device(model, device)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

#%%

# Weights
w= [1 - n / sum(n)]

# Loss Function Train
alpha = 0.05
gamma = 3
focal = kr.losses.FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

train_loop(model, focal, epochs=1200, lr=1e-5, wd=1e-4)

#%% Balanced accuracy
def balanced_accuracy(weights, test_dl, y_eval, model, print_flag=False):
    preds = []
    with torch.no_grad():
        for x1,x2,y in test_dl:
            out = model(x1, x2)*torch.tensor(weights)
            # print(prob)
            preds.append(out)

    y_pred = [torch.argmax(item).item() for sublist in preds for item in sublist]  
    if print_flag: 
        print(accuracy_score(y_eval, y_pred),balanced_accuracy_score(y_eval, y_pred))
        print(confusion_matrix(y_eval, y_pred))
    return -balanced_accuracy_score(y_eval, y_pred)

#%% Train accuracy
train_ds = DisasterPreparednessDataset(X_train, y_train, embedded_col_names)
train_dl = DataLoader(train_ds, batch_size=batch_size)
balanced_accuracy([1,1,1], train_dl, y_train, model, True)

#%% Validation accuracy
val_ds = DisasterPreparednessDataset(X_val, y_val, embedded_col_names)
val_dl = DataLoader(val_ds, batch_size=batch_size)

balanced_accuracy([1,1,1],  val_dl, y_val, model, True)

w_opt=minimize(balanced_accuracy, x0=[1,1,1], args=(val_dl, y_val, model), method='Powell')
balanced_accuracy(w_opt.x, val_dl, y_val, model, True)

#%% Test output
test_ds = DisasterPreparednessDataset(X_test, y_test, embedded_col_names)
test_dl = DataLoader(test_ds, batch_size=batch_size)

balanced_accuracy([1,1,1], test_dl, y_test, model, True)
balanced_accuracy(w_opt.x, test_dl, y_test, model, True)

