#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

print("Loading training data.")
with open('../data/b5_trmatrix.p', 'rb') as ha:
    trdat = pickle.load(ha)

print("Loading attack data.")
with open('../data/b5_atmatrix.p', 'rb') as ha:
    atdat = pickle.load(ha)

trdat = trdat[:, 0:2]
atdat = atdat[:, 0:2]




plt.title('Scatterplot of two principal directions.')
plt.scatter(trdat[:, 0], trdat[:, 1])
plt.scatter(atdat[:, 0], atdat[:, 1], c='red')
plt.show()
