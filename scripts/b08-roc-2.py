#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves

import numpy as np
import matplotlib.pyplot as plt


# kmc eucl - adduser

acc1 = np.array([1, 0.813186813187, 0.681318681319,
    0.527472527473, 0.373626373626, 0.131868131868, 0.032967032967,
    0, 0, 0])
fpr1 = np.array([0.995425434584, 0.776989935956, 0.296889295517,
    0.17886550777, 0.0981244281793, 0.0322506861848, 0.0114364135407,
    0.00411710887466, 0.00297346752059, 0.000228728270814])

# kmc eucl - hydra ftp

acc2 = np.array([1, 0.888888888889, 0.567901234568, 0.308641975309,
    0.191358024691, 0.0617283950617, 0, 0, 0, 0])
# fpr2 = fpr1

# kmc eucl - hydra ssh

acc3 = np.array([0.994318181818, 0.630681818182, 0.357954545455,
    0.227272727273, 0.0852272727273, 0, 0, 0, 0, 0])



# This is the ROC curve

plt.plot(fpr1, acc1, 'r', label='adduser')
plt.plot(fpr1, acc2, 'g', label='hydra ftp')
plt.plot(fpr1, acc3, 'b', label='hydra ssh')
plt.plot(fpr1, fpr1, 'k', label='random')
plt.legend(loc='upper left')
plt.title('k means clustering with euclidean distance')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.show() 


