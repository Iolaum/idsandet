#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PCA on training and attack data!

from __future__ import division
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from os.path import exists


print("Loading data.")

trmatrix = np.load('../data/b1_trmatrix.npy')
atmatrix = np.load('../data/b2_atmatrix.npy')
vamatrix = np.load('../data/b3_vamatrix.npy')


if exists('../data/a13_pca.p'):
    print("Loading pca model.")
    with open('../data/a13_pca.p', 'rb') as ha:
        pca = pickle.load(ha)
else:
    print("Starting PCA on training and attack set!")

    tmatrix = np.concatenate((trmatrix, atmatrix), axis=0)
    tmatrix = np.concatenate((tmatrix, vamatrix), axis=0)
    print("Data combined to matrix {}".format(tmatrix.shape))

    # set up pca
    # keep 80% of variance like in the paper
    pca = PCA(n_components=0.8)

    print("Running PCA algorithm!")

    # train pca
    pca.fit(tmatrix)

    with open('../data/a13_pca.p', 'wb') as ha:
        pickle.dump(pca, ha)

    print("PCA model trained and saved!")

    # clear memory
    del(tmatrix)


print("Principal Component Analysis completed.\n\
The directions of maximum variance are: {}\n\
The variance of each direction is {}".format(pca.components_,
                                             pca.explained_variance_ratio_))

# display variance !?
#var1 = str(pca.explained_variance_ratio_[0])[0:5]
#var2 = str(pca.explained_variance_ratio_[1])[0:5]


# transform data
trdat = pca.transform(trmatrix)
atdat = pca.transform(atmatrix)
vadat = pca.transform(vamatrix)

# prune data to keep 2d
trdat = trdat[:, 0:2]
atdat = atdat[:, 0:2]
vadat = vadat[:, 0:2]

# find min and max
xmin = min([np.amin(trdat[:, 0]), np.amin(atdat[:, 0]), np.amin(vadat[:, 0])])
xmax = max([np.amax(trdat[:, 0]), np.amax(atdat[:, 0]), np.amax(vadat[:, 0])])
ymin = min([np.amin(trdat[:, 1]), np.amin(atdat[:, 1]), np.amin(vadat[:, 1])])
ymax = max([np.amax(trdat[:, 1]), np.amax(atdat[:, 1]), np.amax(vadat[:, 1])])

# plot at + tr
plt.title('Scatterplot of principal directions.')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.scatter(trdat[:, 0], trdat[:, 1], c='green', marker = "+")
plt.scatter(atdat[:, 0], atdat[:, 1], c='red', marker = "+")
plt.savefig('../pictures/a13-pcaplot-1.eps')

# plot at + tr + va
plt.scatter(vadat[:, 0], vadat[:, 1], c='blue', marker = "+")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.savefig('../pictures/a13-pcaplot-2.eps')
plt.close()
