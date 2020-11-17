#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:08:16 2020

@author: Aaron
"""

from feature_extraction import feature_extraction
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


X, y, X_encode, X_train, y_train, X_val, y_val, X_test, y_test, n = \
    feature_extraction(dataset = 0, onehot_option = False, smote_option = False)
            
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

labels = ['1: Relatives/Friends','2: Shelter','3: Hotel']

# Plot transformed data
from matplotlib.colors import LinearSegmentedColormap

colors = [(0.89, 0.36, 0.23),(0.26, 0.24, 0.25),(0.47, 0.5 , 0.71)]
mapML = LinearSegmentedColormap.from_list('229', colors, N = 3)
# colors = ['#E45C3A','#433E3F','#7880B5']
plotcols = [colors[i] for i in y]
fig,ax = plt.subplots(dpi=300, figsize = (6,4))
ax.set_xlabel('Principal Component 1',fontsize=12)
ax.set_ylabel('Principal Component 2',fontsize=12)
ax.set_xticks(np.arange(-8,10,2))
ax.set_yticks(np.arange(-6,16,2))
ax.tick_params(labelsize=12)
# plt.title('2 Component PCA', fontsize = 16)
scatter = ax.scatter(principalDf['PC1'], principalDf['PC2'], c = y, cmap = mapML,
            s = 7)
# scatter = ax.scatter(principalDf['PC1'], principalDf['PC2'], c = plotcols, s = 7)
lgd = ax.legend((scatter.legend_elements()[0]),labels, title="Class",
                    prop={'size': 10}, bbox_to_anchor=(1.01, 1),
                    loc='upper left', ncol=1)
ax.add_artist(lgd)
plt.setp(lgd.get_title(),fontsize=10)
# plt.colorbar()#.set_label(p_names[target])
# plt.tight_layout(rect=(0, 0, -5, 1)) 
plt.gcf().canvas.draw()
invFigure = plt.gcf().transFigure.inverted()
lgd_pos = lgd.get_window_extent()
lgd_coord = invFigure.transform(lgd_pos)
lgd_xmax = lgd_coord[1, 0]
ax_pos = plt.gca().get_window_extent()
ax_coord = invFigure.transform(ax_pos)
ax_xmax = ax_coord[1, 0]
shift = 1 - (lgd_xmax - ax_xmax)
plt.gcf().tight_layout(rect=(0, 0, shift, 1))
plt.show()
fig.savefig('PCA.png',dpi=300)
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
top10_scaledvar = rank_scaledvar[:10]


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