#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def sim_hist(data, bins=25, color='#066170', filename=None):
    fig, ax = plt.subplots()
    n, bins, _ = ax.hist(data, bins, facecolor=color, alpha=0.75)
    ax.vlines(np.mean(data), 0, max(n), colors=color, linestyles='solid', label='mean')
    ax.vlines([np.mean(data) - np.std(data), np.mean(data) + np.std(data)], 0, max(n), colors=color,
              linestyles='dashed', label='std')
    ax.text(0, 0.05 * max(n), "%i datapoints" % len(data), {'ha': 'right', 'va': 'bottom'}, rotation=90)
    ax.text(np.mean(data), 1.01 * max(n), "%.3f +/- %.3f" % (np.mean(data), np.std(data)), {'ha': 'center'})
    ax.set_xlabel('Similarity', fontweight='bold')
    ax.set_ylabel('Counts', fontweight='bold')
    ax.set_title('Pairwise Similarities', fontsize=16, fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def pca_plot(data, reference=None, colors=None, filename=None):
    cut = None
    if not colors:
        colors = ['#066170', '#FDBC1C']
    fig, ax = plt.subplots()
    if len(reference):
        cut = len(data)
        pca = PCA(n_components=2)
        X = pca.fit_transform(np.vstack((data, reference)))
        ax.plot(X[cut:, 0], X[cut:, 1], 'o', c=colors[1], label='reference')
    else:
        pca = PCA(n_components=2)
        X = pca.fit_transform(data)

    ax.plot(X[:cut, 0], X[:cut, 1], '*', c=colors[0], label='data')

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
