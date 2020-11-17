#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:45:43 2020

@author: Davyd, Ana, Aaron
"""
from feature_extraction import feature_extraction
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import seaborn as sns

torch.manual_seed(1)
np.random.seed(0)

# dataset: 0 = SF data only, 1 = SF + LA data, 2 = SF + SJ data, 3 = All of CA

X, y, X_encode, X_train, y_train, X_val, y_val, X_test, y_test, n = \
    feature_extraction(dataset = 0, onehot_option = False, smote_option = False)
    
    
#%% Categorical embedding for categorical columns having more than two values

# Choosing columns for embedding
embedded_cols = {n: len(col.cat.categories) for n,col in X.loc[:,X_encode==1].items() if len(col.cat.categories) > 2}
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
        D2 = 2*(self.n_emb + self.n_cont)//3 + 3
        D3 = 3
        self.lin1 = nn.Linear(D1, D2) #just CS things
        self.lin2 = nn.Linear(D2, D3)
        self.bn1 = nn.BatchNorm1d(self.n_cont) # n_cont = number of cont. features
        self.bn2 = nn.BatchNorm1d(D2)
        self.emb_drop = nn.Dropout(0.1) # dropout probability for features
        self.drops = nn.Dropout(0.5) # dropout probability for hidden layers


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
        
        return x

#%% More function definition

# Optimizer
def get_optimizer(model, lr = 0.001, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim

# Training function
def train_model(model, optim, train_dl, w):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0] # size of batch
        output = model(x1, x2) # forward pass
        loss = F.cross_entropy(output, y, weight=torch.tensor(w).float())
        optim.zero_grad() #don't accumulate gradients in the optimizer object
        loss.backward() # calculate gradient (backward pass)
        optim.step() # take gradient descent step
        total += batch # add batch loss to total loss
        sum_loss += batch*(loss.item())
    return sum_loss/total

# Evaluation function
def val_loss(model, valid_dl, w):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    pred_out = []
    y_out = []
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = F.cross_entropy(out, y, weight=torch.tensor(w).float())
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        pred_out = np.hstack((pred_out,np.asarray(pred)))
        y_out = np.hstack((y_out,np.asarray(y)))
        correct += (pred == y).float().sum().item()
    print("valid loss %.3f, total accuracy %.3f, and balanced accuracy %.3f" % 
          (sum_loss/total, correct/total, balanced_accuracy_score(y_out, pred_out)))
    
    return sum_loss/total, correct/total

def train_loop(model, w, epochs, lr=0.01, wd=0.0):
    optim = get_optimizer(model, lr = lr, wd = wd)
    for i in range(epochs): 
        loss = train_model(model, optim, train_dl, w)
        # print("training loss: ", loss)
        val_loss(model, valid_dl, w)

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

#%% Train

# Weights
w= [1 - n / sum(n)]

train_loop(model, w, epochs=250, lr=1e-4, wd=1e-2)


#%%
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
        print(f1_score(y_eval, y_pred, average='weighted'))
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


test_ds = DisasterPreparednessDataset(X_test, y_test, embedded_col_names)
test_dl = DataLoader(test_ds, batch_size=batch_size)


#%%
test_ds = DisasterPreparednessDataset(X_test, y_test, embedded_col_names)
test_dl = DataLoader(test_ds, batch_size=batch_size)

preds = []
with torch.no_grad():
    for x1,x2,y in test_dl:
        out = model(x1, x2)*w_opt.x
        # print(prob)
        preds.append(out)


y_pred = [torch.argmax(item).item() for sublist in preds for item in sublist]  

#%%
import matplotlib.pyplot as plt
def plot_multiclass_roc(preds, y_pred, X_test, y_test, n_classes, title, figsize=(5,9.5), flag=False, save=None):
    y_score = torch.cat(preds,dim=0)
        
    colors = ['#433E3F','#7880B5', '#E45C3A']
    
    plt.rcParams['font.size'] = '16'

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_aspect(1)
    ax.grid('on')
    ax.minorticks_on()
    ax.grid(b=True, which='major', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(b=True, which='minor',  color='gray', linestyle='-', linewidth=0.25, alpha=0.25, zorder=0)

    ax.set_title(title, fontsize=18, fontweight='bold')
    titles = ['Relatives / Friends', 'Public Shelter', 'Hotel']
    for i in range(n_classes):
        print('ROC curve (area = %0.4f) for label %i' % (roc_auc[i], i))
        # ax.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
        ax.plot(fpr[i], tpr[i], color=colors[i], label=f'Class {i+1}', linewidth=3)
    ax.legend(loc="lower right")
    
    np.set_printoptions(precision=2)
    
    ax2 = sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True, 
                      cmap=plt.cm.Blues, vmin=0.0, vmax=1.0, annot_kws={'size':16},
                      yticklabels=[i+1 for i in range(n_classes)],
                      xticklabels=[i+1 for i in range(n_classes)])

    for _, spine in ax2.spines.items():
        spine.set_visible(True)
        
    ax2.set_xlabel('Predicted class')
    ax2.set_ylabel('True class')
    ax2.set_aspect(1)

    fig.tight_layout()
    
    if save: plt.savefig(save, dpi=300)
    
    plt.show()
   
#%%
plot_multiclass_roc(preds, y_pred, X_test, y_test, title='Neural Network (W. CE)', n_classes=3, flag=False, save='NN_WCE_roc.png')
