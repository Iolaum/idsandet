#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

from __future__ import division
import pickle
import numpy as np
from sklearn.decomposition import PCA


print("Starting PCA on training set!")


with open('../data/b1_trmatrix.p', 'rb') as ha:
    tmatrix = pickle.load(ha)


# set up pca
# keep 80% of variance like in the paper
pca = PCA(n_components=0.8)

print("Running PCA algorithm!")

pca.fit(tmatrix)

print("Principal Component Analysis completed.\n\
The directions of maximum variance are: {}\n\
The variance of each direction is {}".format(pca.components_,
                                             pca.explained_variance_ratio_))

with open('../data/b4_pca.p', 'wb') as ha:
    pickle.dump(pca, ha)

print("PCA model trained and saved!")
