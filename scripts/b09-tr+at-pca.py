#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PCA on training and attack data!
# DEPRECATED after refactoring.
# to be deleted !?

from __future__ import division
import pickle
import numpy as np
from sklearn.decomposition import PCA


print("Starting PCA on training and attack set!")


with open('../data/b1_trmatrix.p', 'rb') as ha:
    trmatrix = pickle.load(ha)

with open('../data/b2_atmatrix.p', 'rb') as ha:
    atmatrix = pickle.load(ha)


tmatrix = np.concatenate((trmatrix, atmatrix), axis=0)


# set up pca
# keep 80% of variance like in the paper
pca = PCA(n_components=0.8)

print("Running PCA algorithm!")

pca.fit(tmatrix)

print("Principal Component Analysis completed.\n\
The directions of maximum variance are: {}\n\
The variance of each direction is {}".format(pca.components_,
                                             pca.explained_variance_ratio_))

with open('../data/b9_pca.p', 'wb') as ha:
    pickle.dump(pca, ha)

print("PCA model trained and saved!")


# 11 dimensions for 80% variance
# meaning in the paper they used only training set !!
