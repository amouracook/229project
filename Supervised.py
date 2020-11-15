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


# dataset: 0 = SF data only, 1 = SF + LA data, 2 = SF + SJ data, 3 = All of CA

X, X_encode, X_train, y_train, X_val, y_val, X_test, y_test, n = \
    feature_extraction(dataset = 0, onehot_option = False, smote_option = True)

    
#%% Ridge regression classifier
from sklearn.linear_model import RidgeClassifier
''' NOTE: works better with label encoding instead of one hot '''

clf = RidgeClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

print(clf.score(X_val, y_val))
print(balanced_accuracy_score(y_val, clf.predict(X_val)))
print(confusion_matrix(y_val , clf.predict(X_val)))

#%% XGBoost

from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree

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


