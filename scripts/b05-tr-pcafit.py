#!/usr/bin/env python
# -*- coding: utf-8 -*-

# See all training files and create a
# training data dictionary of lists.

from __future__ import division
import pickle
import numpy as np
from sklearn.decomposition import PCA


print("Loading PCA model.")

with open('../data/b4_pca.p', 'rb') as ha:
    pca = pickle.load(ha)

print("Started processing the data set.")

# # training data
tmatrix = np.load("../data/b1_trmatrix.npy")

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving training data principal components: {}".format(ntmatrix.shape))

np.save("../data/b5_trmatrix", ntmatrix)


# # attack data
tmatrix = np.load('../data/b2_atmatrix.npy')

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving attack data principal components: {}".format(ntmatrix.shape))

np.save("../data/b5_atmatrix", ntmatrix)


# # validation data
tmatrix = np.load("../data/b3_vamatrix.npy")

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving validation data principal components: {}".format(ntmatrix.shape))

np.save("../data/b5_vamatrix", ntmatrix)


# # adduser attack
tmatrix = np.load("../data/b2_at1mat_adduser.npy")

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving adduser attack data principal components: {}".format(ntmatrix.shape))

np.save('../data/b5_at1mat_adduser', ntmatrix)

# # hydra ftp attack
tmatrix = np.load('../data/b2_at2mat_hyftp.npy')

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving hydra ftp attack data principal components: {}".format(ntmatrix.shape))

np.save('../data/b5_at2mat_hyftp', ntmatrix)


# # hydra ssh attack
tmatrix = np.load("../data/b2_at3mat_hyssh.npy")

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving hydra ssh attack data principal components: {}".format(ntmatrix.shape))

np.save('../data/b5_at3mat_hyssh', ntmatrix)


# # java meterpreter attack
tmatrix = np.load("../data/b2_at4mat_javamet.npy")

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving java meterpreter attack data principal components: {}".format(ntmatrix.shape))

np.save('../data/b5_at4mat_javamet', ntmatrix)


# # meterpreter attack
tmatrix = np.load("../data/b2_at5mat_meter.npy")

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving meterpreter attack data principal components: {}".format(ntmatrix.shape))

np.save('../data/b5_at5mat_meter', ntmatrix)


# # web shell attack
tmatrix = np.load("../data/b2_at6mat_webshell.npy")

# perform pca
ntmatrix = pca.transform(tmatrix)

print("Saving web shell attack data principal components: {}".format(ntmatrix.shape))

np.save('../data/b5_at6mat_webshell', ntmatrix)
