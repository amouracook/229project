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
top10_scaledvar = rank_scaledvar[:11]

#%% Plot principal components (3-pane)

from matplotlib.colors import LinearSegmentedColormap
labels = ['Relative/Friend','Shelter','Hotel']
colors = ['#E45C3A','#433E3F','#7880B5']
mapML = LinearSegmentedColormap.from_list('229', colors, N = 3)
plotcols = [colors[i] for i in y]

# Create 3 pane figure
fig,(ax1,ax2,ax3) = plt.subplots(1, 3, dpi=300, figsize = (11,4), sharex=True, sharey=True)
plt.xticks(np.arange(-8,10,4))
plt.yticks(np.arange(-6,16,4))
fig.text(0, 0.5, 'Principal Component 2', va='center', rotation='vertical', fontsize=16)

# Loop through and plot one PCA on each subplot   
axs = [ax1,ax2,ax3] 
for i in range(len(axs)):
    ax = axs[i]
    # ax.set_aspect(1)
    ax.tick_params(labelsize=16)
    ax.grid(b=True, which='major', linestyle='-', linewidth=0.5, alpha=0.25, zorder=0)
    ax.scatter(principalDf.loc[y==i]['PC1'], principalDf.loc[y==i]['PC2'], c=colors[i], s = 14, edgecolors='k', linewidth=0.3, label=f'Class {i+1}: {labels[i]}')

# Create and place legends for each subplot
plt.axes(axs[0])
ax1.legend(bbox_to_anchor=(1.05,1.15),frameon=False,fontsize=15)
plt.axes(axs[1])
ax2.legend(bbox_to_anchor=(0.93,1.15),frameon=False,fontsize=15)
plt.axes(axs[2])
ax3.legend(bbox_to_anchor=(0.87, 1.15),frameon=False,fontsize=15)
ax2.set_xlabel('Principal Component 1', ha='center', fontsize=16)

# Save figure without cropping out external legend
plt.gcf().canvas.draw()
invFigure = plt.gcf().transFigure.inverted()
lgd_pos = fig.get_window_extent()
lgd_coord = invFigure.transform(lgd_pos)
lgd_xmax = lgd_coord[1, 0]
ax_pos = plt.gca().get_window_extent()
ax_coord = invFigure.transform(ax_pos)
ax_xmax = ax_coord[1, 0]
shift = 1 - (lgd_xmax - ax_xmax)
plt.gcf().tight_layout(rect=(0, 0, shift, 1))
plt.show()
fig.savefig('PCA.png',dpi=300)

#%% Plot principal components (1-pane)


a = plt.figure(dpi=300)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
from matplotlib.colors import LinearSegmentedColormap

# Define custom colors
colors = [(0.89, 0.36, 0.23),(0.26, 0.24, 0.25),(0.47, 0.5 , 0.71)]
mapML = LinearSegmentedColormap.from_list('229', colors, N = 3)
plotcols = [colors[i] for i in y]

# Create figure
fig,ax = plt.subplots(dpi=300, figsize = (6,4))
ax.set_xlabel('Principal Component 1',fontsize=12)
ax.set_ylabel('Principal Component 2',fontsize=12)
ax.set_xticks(np.arange(-8,10,2))
ax.set_yticks(np.arange(-6,16,2))
ax.tick_params(labelsize=12)
plt.scatter(principalDf['PC1'], principalDf['PC2'], c = y,cmap='viridis', s = 4)
plt.legend()

# Plot colored scatter plot with all three classes
scatter = ax.scatter(principalDf['PC1'], principalDf['PC2'], c = y, cmap = mapML,
            s = 7)
lgd = ax.legend((scatter.legend_elements()[0]),labels, title="Class",
                    prop={'size': 10}, bbox_to_anchor=(1.01, 1),
                    loc='upper left', ncol=1)

# Save figure without cropping out external legend
ax.add_artist(lgd)
plt.setp(lgd.get_title(),fontsize=10)
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
a.savefig('PCA.png',dpi=300)
fig.savefig('PCA.png',dpi=300)
plt.close()
