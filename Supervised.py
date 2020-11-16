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

#%% Defined multiclass roc function
def plot_multiclass_roc(clf, X_test, y_test, n_classes, title, figsize=(12,4), flag=False):
    if flag: y_score = clf.predict_proba(X_test)
    else: y_score = clf.decision_function(X_test)
    
    colors = ['#E45C3A', '#F4A261', '#7880B5']

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
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid('on')
    ax.minorticks_on()
    ax.grid(b=True, which='major', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(b=True, which='minor',  color='gray', linestyle='-', linewidth=0.25, alpha=0.25, zorder=0)

    # ax.set_title('Receiver operating characteristic')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    
    np.set_printoptions(precision=2)
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=['label '+str(i) for i in range(n_classes) ],
                                 cmap=plt.cm.Blues,
                                 ax=ax2,
                                 normalize='true')
    fig.tight_layout()
    plt.show()
    fig.savefig(f'{title}.png', dpi=300)
    
#%% Load dataset
# dataset: 0 = SF data only, 1 = SF + LA data, 2 = SF + SJ data, 3 = All of CA

X, X_encode, X_train, y_train, X_val, y_val, X_test, y_test, n = \
    feature_extraction(dataset = 0, onehot_option = False, smote_option = True, as_category=False)

    
#%% Ridge regression classifier
from sklearn.linear_model import RidgeClassifier

ridge = RidgeClassifier(class_weight='balanced')
ridge.fit(X_train, y_train)

print('RIDGE REGRESSION')
# Validation
print(ridge.score(X_val, y_val))
print(balanced_accuracy_score(y_val, ridge.predict(X_val)))
print(f1_score(y_val, ridge.predict(X_val), average='macro'))
print(confusion_matrix(y_val , ridge.predict(X_val)))

# Test
print(ridge.score(X_test, y_test))
print(balanced_accuracy_score(y_test, ridge.predict(X_test)))
print(f1_score(y_test, ridge.predict(X_test), average='macro'))
print(confusion_matrix(y_test , ridge.predict(X_test)))


plot_multiclass_roc(ridge, X_test, y_test, title='ROCRidge', n_classes=3, flag=False)

#%% XGBoost
from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree

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
print(f1_score(y_val, y_pred, average='macro'))
print(confusion_matrix(y_val, y_pred))


# Test
y_pred = xgb.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))
print(confusion_matrix(y_test , y_pred))


plot_multiclass_roc(xgb, X_test, y_test, title='ROCXGBoost', n_classes=3, flag=True)

#%% Importance graph
fig, ax = plt.subplots(1, 1, dpi=300)
fig.set_size_inches(8,5)

ax.minorticks_on()

plot_importance(xgb, max_num_features=15, color='k', grid=False, ax=ax, zorder=10) # top 10 most important features

ax.grid(b=True, which='major', axis='x', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
ax.grid(b=True, which='minor', axis='x', color='gray', linestyle='-', linewidth=0.25, alpha=0.25, zorder=0)


fig.tight_layout()

plt.savefig('xgboost_feature_importance.png', dpi=300)

plt.show()

#%%
# plot_tree(model, num_trees=0, dpi=300)
# plt.show()

#%% Convergence plot

# # retrieve performance metrics
# results = model.evals_result()
# epochs = len(results['validation_0']['merror'])
# x_axis = range(0, epochs)

# fig, ax = plt.subplots(1, 1, dpi=300)
# fig.set_size_inches(8,5)
# ax.plot(x_axis, results['validation_0']['merror'], label='Train')
# ax.plot(x_axis, results['validation_1']['merror'], label='Validation')
# ax.legend()
# pyplot.ylabel('Classification Error')
# pyplot.title('XGBoost Classification Error')
# pyplot.show()

#%%
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
print(f1_score(y_val, y_pred, average='macro'))
print(confusion_matrix(y_val, y_pred))


# Test
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))
print(confusion_matrix(y_test , ridge.predict(X_test)))

plot_multiclass_roc(rf, X_test, y_test, title='ROCRandomForest',n_classes=3, flag=True)

#%%
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0, 
                           multi_class='multinomial', 
                           penalty='l2', max_iter=10000)
logreg.fit(X_train, y_train)

print('LOGREG')

# Validation
y_pred = logreg.predict(X_val)
print(accuracy_score(y_val, y_pred))
print(balanced_accuracy_score(y_val, y_pred))
print(f1_score(y_val, y_pred, average='macro'))
print(confusion_matrix(y_val, y_pred))


# Test
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='macro'))
print(confusion_matrix(y_test , ridge.predict(X_test)))

plot_multiclass_roc(logreg, X_test, y_test, title='ROCLogReg', n_classes=3,  flag=False)


