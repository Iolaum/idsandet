#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

from __future__ import division
import pickle
import numpy as np
from sklearn.decomposition import PCA


print("Starting PCA on training set!")


tmatrix = np.load('../data/b1_trmatrix.npy')

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

'''Starting PCA on training set!
Running PCA algorithm!
Principal Component Analysis completed.
The directions of maximum variance are: [[  1.08090413e-03  -4.85364899e-01  -3.46838852e-01 ...,  -1.84256194e-04
    5.49754510e-06  -1.53284048e-04]
 [ -6.95007310e-04   8.40335064e-02   8.36267956e-01 ...,   5.70704291e-04
   -1.07656277e-06  -6.47492024e-04]
 [  1.12577068e-03  -1.64799573e-01   8.82030789e-02 ...,  -7.00812827e-04
   -9.75261248e-06   2.91719370e-05]
 ..., 
 [  1.37187368e-03  -9.74944800e-02   1.46480176e-02 ...,   2.35746730e-03
    7.00829006e-05  -2.78310361e-04]
 [ -1.56129334e-06  -1.25409247e-02  -2.75117926e-02 ...,  -1.40262115e-03
   -1.09385686e-05  -3.00071017e-03]
 [  4.66736000e-03  -1.61048396e-02  -5.22472496e-02 ...,   4.91734618e-04
    3.61914840e-05   5.52419588e-04]]
The variance of each direction is [ 0.1969005   0.15480881  0.11299774  0.09653481  0.07091701  0.06885015
  0.05168454  0.03804072  0.02933048]
PCA model trained and saved!
'''
