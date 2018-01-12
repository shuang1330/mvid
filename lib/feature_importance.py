#!/usr/bin/env python

"""
visualize feature importance of a model consists of
a feature extractor (pca only currently)
and a classifier (only linear functions,
                  no tree-based methods currently)
"""

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

# draw a overlapping bar graph for pandas dataframes
def overlapped_bar(df, show=False, width=0.9, alpha=.3,
                   title='', xlabel='', ylabel='', **plot_kwargs):
    """draw a stacked bar chart except bars on top of
    each other with transparency"""
    pl.figure(figsize=(20, 10))
    xlabel = xlabel or df.index.name
    N = len(df)
    M = len(df.columns)
    indices = np.arange(N)
    colors = ['steelblue','firebrick',
              'darksage','goldenrod','gray'] * int(M / 5. + 1)
    for i, label, color in zip(range(M), df.columns, colors):
        kwargs = plot_kwargs
        kwargs.update({'color': color, 'label': label})
        plt.bar(indices, df[label], width=width,
                alpha=alpha if i else 1, **kwargs)
        plt.xticks(indices + .5 * width,
                   ['{}'.format(idx) for idx in df.index.values])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    if show:
        plt.show()
    return plt.gcf()

def display_feature_importance(mean_x,x_cls,pca,classifier,LOG_TRANS=True,
                       TABLE=False,DRAW=False,SAVE_DRAW=False):
    '''
    visualize feature importance of a model consists of
    a feature extractor and a classifier by
    drawing/returning the coefficients.
    Input: mean_x, list of mean values of different classes (at least 2)
           x_cls, list of names of the classes
           pca, sklearn.Decomposition.PCA()
           classifier, sklearn models with .coef_ attribute
           LOG_TRANS, transform the importances in log-scale
           TABLE, return the importance value of each features
           DRAW, plot the importances
           SAVE_DRAW, save the plot as 'importances.png'
    Output: importances, list of importance values for each feature per class
            plot of feature importances
    '''
    importances = []
    length = pca.components_.shape[0]
    if classifier.coef_ is not None:
        importance = pca.components_*\
        classifier.coef_.reshape((length,1))
    elif classifier.feature_importance_:
        importance = pca.components_*\
        classifier.feature_importances_.reashape((length,1))
    for cls_mean in mean_x:
        if LOG_TRANS:
            importances.append(np.log(abs(importance.sum(axis=0)*\
                                          cls_mean)))
        else:
            importances.append(abs(importance.sum(axis=0)*cls_mean))
    if DRAW:
        df = pd.DataFrame(np.matrix(importances).T, columns=x_cls,
                  index=mean_x[0].keys())
        overlapped_bar(df, show=True)
        plt.show()
        if SAVE_DRAW:
            plt.savefig('importances.png')
    if TABLE:
        return importance
