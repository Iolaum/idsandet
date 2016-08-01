#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PCA on training and attack data!

from __future__ import division
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


print("Starting PCA on training and attack set!")


with open('../data/b1_trmatrix.p', 'rb') as ha:
    trmatrix = pickle.load(ha)

with open('../data/b2_atmatrix.p', 'rb') as ha:
    atmatrix = pickle.load(ha)

with open('../data/b3_vamatrix.p', 'rb') as ha:
    vamatrix = pickle.load(ha)

with open('../data/b9_pca.p', 'rb') as ha:
    pca = pickle.load(ha)

trdat = pca.transform(trmatrix)
atdat = pca.transform(atmatrix)
vadat = pca.transform(vamatrix)

trdat = trdat[:, 0:2]
atdat = atdat[:, 0:2]
vadat = vadat[:, 0:2]

# plot at + tr

plt.title('Scatterplot of two principal directions.')
plt.xlabel('p1')
plt.ylabel('p2')
plt.scatter(trdat[:, 0], trdat[:, 1], c='green')
plt.scatter(atdat[:, 0], atdat[:, 1], c='red')
plt.savefig('../pictures/a13-pcaplot-1.jpeg')

# plot at + tr + va

plt.title('Scatterplot of two principal directions.')
plt.xlabel('p1')
plt.ylabel('p2')
plt.scatter(trdat[:, 0], trdat[:, 1], c='green')
plt.scatter(atdat[:, 0], atdat[:, 1], c='red')
plt.scatter(vadat[:, 0], vadat[:, 1], c='blue')
plt.savefig('../pictures/a13-pcaplot-2.jpeg')
