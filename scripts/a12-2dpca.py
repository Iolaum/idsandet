#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Plot 2 principal Components from training set

import pickle
import numpy as np
import matplotlib.pyplot as plt

print("Loading training data.")
trdat = np.load('../data/b5_trmatrix.npy')


print("Loading attack data.")
atdat = np.load('../data/b5_atmatrix.npy')


trdat = trdat[:, 0:2]
atdat = atdat[:, 0:2]




plt.title('Scatterplot of two principal directions.')
plt.scatter(trdat[:, 0], trdat[:, 1])
plt.scatter(atdat[:, 0], atdat[:, 1], c='red')
plt.show()
