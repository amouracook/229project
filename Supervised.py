#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:17:20 2020

@author:  Davyd, Ana, Aaron
"""

from feature_extraction import feature_extraction
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pylab as plt
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn import preprocessing, model_selection
from imblearn import over_sampling as os


#%% Defined multiclass ROC plot function
def plot_multiclass_roc(clf, X_test, y_test, n_classes, title, figsize=(6.5,12), flag=False, save=None):
    if flag: y_score = clf.predict_proba(X_test)
    else: y_score = clf.decision_function(X_test)
        
    colors = ['#7880B5','#433E3F', '#E45C3A']
    # colors = ['#2D5362', '#7880B5', '#79A1CC', '#2A9D8F', '#E9C46A', '#F4A261', '#E45C3A']
    
    plt.rcParams['font.size'] = '23'

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate dummies once
    y_temp = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_temp[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ROC for each class
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect(1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid('on')
    ax.minorticks_on()
    ax.grid(b=True, which='major', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(b=True, which='minor',  color='gray', linestyle='-', linewidth=0.25, alpha=0.25, zorder=0)

    # ax.set_title(title, fontsize=20, fontweight='bold')
    titles = ['Relatives / Friends', 'Public Shelter', 'Hotel']
    for i in range(n_classes):
        print('ROC curve (area = %0.4f) for label %i' % (roc_auc[i], i))
        ax.plot(fpr[i], tpr[i], color=colors[i], label=f'Class {i+1}', linewidth=3)
    ax.legend(loc="lower right", fontsize=20)
    
    np.set_printoptions(precision=2)
    ax2 = sns.heatmap(confusion_matrix(y_test, clf.predict(X_test), normalize='true'), annot=True, 
                      cmap=plt.cm.Blues, vmin=0.0, vmax=1.0, annot_kws={'size':23},
                      yticklabels=[i+1 for i in range(n_classes)],
                      xticklabels=[i+1 for i in range(n_classes)])
    
    for _, spine in ax2.spines.items():
        spine.set_visible(True)
        
    ax2.set_xlabel('Predicted label')
    ax2.set_ylabel('True label')
    ax2.set_aspect(1)

    fig.tight_layout()

    if save: plt.savefig(save, dpi=300)
    
    plt.show()
    
#%% Load dataset
# dataset: 0 = SF data only, 1 = SF + LA data, 2 = SF + SJ data, 3 = All of CA

X, y, X_encode, X_train, y_train, X_val, y_val, X_test, y_test, n = \
    feature_extraction(dataset = 0, onehot_option = False, smote_option = True, as_category=False)


#%% Ridge regression classifier
from sklearn.linear_model import RidgeClassifier

ridge = RidgeClassifier(alpha=0.1)
ridge.fit(X_train, y_train)

print('RIDGE REGRESSION')
# Validation
print(ridge.score(X_val, y_val))
print(balanced_accuracy_score(y_val, ridge.predict(X_val)))
print(f1_score(y_val, ridge.predict(X_val), average='weighted'))
print(confusion_matrix(y_val , ridge.predict(X_val)))

# Test
print(ridge.score(X_test, y_test))
print(balanced_accuracy_score(y_test, ridge.predict(X_test)))
print(f1_score(y_test, ridge.predict(X_test), average='weighted'))
print(confusion_matrix(y_test , ridge.predict(X_test)))

# Plot ROC curves and confusion matrix
plot_multiclass_roc(ridge, X_test, y_test, title='Ridge Regression', n_classes=3, flag=False)

# Define parameters of interest
feature_names = {'HHGRAD': 'Education Level',
                  'BLD':'Unit Type',
                  'TOTHCAMT': 'Housing Cost PM',
                  'FINCP': 'Family Income',
                  'HHINUSYR': 'Year Came to US',
                  'HINCP': 'HH Income',
                  'YRBUILT': 'Year Build',
                  'HHAGE': 'HH Age',
                  'UTILAMT': 'Utility PM',
                  'INSURAMT': 'Insurance PM',
                  'MARKETVAL': 'Unit Market Value',
                  'HHMOVE': 'Year Moved In',
                  'UNITSIZE': 'Unit Size',
                  'HHMAR': 'Marital Status',
                  'TOTROOMS': '# of Rooms',
                  'SAMEHHLD': 'HH Unchanged',
                  'RATINGHS': 'Livability Rating',
 }

# Print a coefficient corresponding to each feature in the model
coefs = ridge.coef_
cols = X_train.columns
features = feature_names.keys()

for f in features:
    print(f, coefs[:,X_train.columns.get_loc(f)])

#%% XGBoost classifier
from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer

xgb = XGBClassifier(n_estimators=500, 
                      eta=0.1, 
                      max_depth=3,
                      colsample_bytree=0.1,
                      reg_lambda=5e2,
                      subsample=0.3,
                      random_state=4)

xgb.fit(X_train, y_train, eval_metric=['merror', 'mlogloss'],
          eval_set=[(X_train, y_train), (X_val, y_val)], 
          verbose=False)

print('XGB')

# Validation
y_pred = xgb.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred, average='weighted'))
print(confusion_matrix(y_val, y_pred))

# Test
y_pred = xgb.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))
print(confusion_matrix(y_test , y_pred))

# Plot ROC curves and confusion matrix
plot_multiclass_roc(xgb, X_test, y_test, title='XGBoost', n_classes=3, flag=True)

#%% Convergence plot

# Retrieve model loss history
results = xgb.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

# Plot convergence curves
fig, ax = plt.subplots(1, 1, dpi=300)
fig.set_size_inches(8,5)
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Validation')
ax.legend()
ax.set_ylabel('Classification Error')
ax.set_title('XGBoost Classification Error')
plt.show()

#%% Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=3, 
                               random_state=2,
                               n_estimators=100,
                               min_samples_leaf=3,
                               max_features='sqrt')

rf.fit(X_train, y_train)

print('RF')

# Validation
y_pred = rf.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred, average='weighted'))
print(confusion_matrix(y_val, y_pred))

# Test
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))
print(confusion_matrix(y_test , rf.predict(X_test)))

# Plot ROC curves and confusion matrix
plot_multiclass_roc(rf, X_test, y_test, title='Random Forest',n_classes=3, flag=True)

#%% Feature importance plot for RF and XGBoost models

plt.rcParams['font.size'] = '16'
fig, (ax1,ax2) = plt.subplots(1, 2, dpi=300)
fig.set_size_inches(11.5,4.5)

for ax in [ax1,ax2]:
    ax.minorticks_on()
    ax.set_xticks(range(10))
    
    ax.grid(b=True, which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(b=True, which='minor', axis='y', color='gray', linestyle='-', linewidth=0.25, alpha=0.25, zorder=0)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features=X.columns
ax1.bar(range(10), importances[[indices[i] for i in range(10)]], color='k', align='center',width=0.5)
ax1.set_xticklabels(labels=[feature_names[features[indices[i]]] for i in range(10)], rotation=45, ha='right')
ax1.set_ylabel('Relative Importance')
ax1.set_title('Random Forest', fontsize=16, fontweight='bold')
ax1.set(yticks=np.arange(0,0.15,0.02))


importances = xgb.get_booster().get_score(importance_type="weight")
features=np.asarray(list(importances.keys()))
importances = np.asarray(list(importances.values()))
indices = np.argsort(importances)[::-1]
ax2.bar(range(10), importances[[indices[i] for i in range(10)]], color='k', align='center', width=0.5)
ax2.set_xticklabels(labels=[feature_names[features[indices[i]]] for i in range(10)], rotation=45, ha='right')
ax2.set_ylabel('F Score')
ax2.set_title('XGBoost', fontsize=16, fontweight='bold')
ax2.set(yticks=np.arange(0,350,50))

fig.tight_layout()

# plt.savefig('feature_importance.png', dpi=300)

plt.show()

#%% Softmax Regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0, 
                           multi_class='multinomial', 
                           penalty='l2', 
                           max_iter=10000,
                           C=0.1)
logreg.fit(X_train, y_train)

print('SOFTMAX')

# Validation
y_pred = logreg.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred, average='weighted'))
print(confusion_matrix(y_val, y_pred))

# Test
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))
print(confusion_matrix(y_test , ridge.predict(X_test)))

# Plot ROC curves and confusion matrix
plot_multiclass_roc(logreg, X_test, y_test, title='Softmax Regression', n_classes=3,  flag=False)


