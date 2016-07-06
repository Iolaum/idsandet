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
with open('../data/b1_trmatrix.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving training data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_trmatrix.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)

# # attack data
with open('../data/b2_atmatrix.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving attack data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_atmatrix.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)


# # validation data
with open('../data/b3_vamatrix.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving validation data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_vamatrix.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)

# # adduser attack
with open('../data/b2_at1mat_adduser.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving adduser attack data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_at1mat_adduser.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)

# # hydra ftp attack
with open('../data/b2_at2mat_hyftp.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving hydra ftp attack data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_at2mat_hyftp.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)

# # hydra ssh attack
with open('../data/b2_at3mat_hyssh.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving hydra ssh attack data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_at3mat_hyssh.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)

# # java meterpreter attack
with open('../data/b2_at4mat_javamet.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving java meterpreter attack data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_at4mat_javamet.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)

# # meterpreter attack
with open('../data/b2_at5mat_meter.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving meterpreter attack data principal components: {}".format(ntmatrix.shape))

with open('../data/b5_at5mat_meter.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)

# # web shell attack
with open('../data/b2_at6mat_webshell.p', 'rb') as ha:
    tmatrix = pickle.load(ha)

ntmatrix = pca.transform(tmatrix)

print("Saving web shell attack data principal components: {}".format(ntmatrix.shape))

with open('../data/b6_at6mat_webshell.p', 'wb') as ha:
    pickle.dump(ntmatrix, ha)
