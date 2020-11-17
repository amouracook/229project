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

# colors = [(0.89, 0.36, 0.23),(0.26, 0.24, 0.25),(0.47, 0.5 , 0.71)]
mapML = LinearSegmentedColormap.from_list('229', colors, N = 3)
colors = ['#E45C3A','#433E3F','#7880B5']
plotcols = [colors[i] for i in y]

fig,(ax1,ax2,ax3) = plt.subplots(1, 3, dpi=300, figsize = (8,3.5), sharex=True, sharey=True)
plt.xticks(np.arange(-8,10,2))
plt.yticks(np.arange(-6,16,2))

fig.text(0, 0.5, 'Principal Component 2', va='center', rotation='vertical', fontsize=12)


   
axs = [ax1,ax2,ax3] 
for i in range(len(axs)):
    ax = axs[i]
    ax.set_aspect(1)
    ax.tick_params(labelsize=12)
    ax.grid(b=True, which='major', linestyle='-', linewidth=0.5, alpha=0.25, zorder=0)

    # plt.title('2 Component PCA', fontsize = 16)
    # scatter = ax.scatter(principalDf['PC1'], principalDf['PC2'], c = plotcols, s = 7)
    
ax1.scatter(principalDf.loc[y==0]['PC1'], principalDf.loc[y==0]['PC2'], c=colors[0], s = 7, label=f'Class {1}: Relatives/Friends')
ax2.scatter(principalDf.loc[y==1]['PC1'], principalDf.loc[y==1]['PC2'], c=colors[1], s = 7, label=f'Class {2}: Shelter')
ax3.scatter(principalDf.loc[y==2]['PC1'], principalDf.loc[y==2]['PC2'], c=colors[2], s = 7, label=f'Class {3}: Hotel')

ax2.set_xlabel('Principal Component 1', ha='center', fontsize=12)

# lgd = ax.legend((scatter.legend_elements()[0]),labels, title="Class",
#                         prop={'size': 10}, bbox_to_anchor=(1.01, 1),
#                         loc='upper left', ncol=1)
# ax.add_artist(lgd)
# plt.setp(lgd.get_title(),fontsize=10)
    
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(handles, labels, loc='upper center', fontsize=12, ncol=3, bbox_to_anchor=(0.52, 1.05))

# plt.colorbar()#.set_label(p_names[target])
# plt.tight_layout(rect=(0, 0, -5, 1)) 
# plt.gcf().canvas.draw()
# invFigure = plt.gcf().transFigure.inverted()
# lgd_pos = lgd.get_window_extent()
# lgd_coord = invFigure.transform(lgd_pos)
# lgd_xmax = lgd_coord[1, 0]
# ax_pos = plt.gca().get_window_extent()
# ax_coord = invFigure.transform(ax_pos)
# ax_xmax = ax_coord[1, 0]
# shift = 1 - (lgd_xmax - ax_xmax)
# plt.gcf().tight_layout(rect=(0, 0, shift, 1))
fig.tight_layout()
plt.show()
fig.savefig('PCA.png',dpi=300)
# plt.close()

#%% Get component loadings 
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