#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves

import numpy as np
import matplotlib.pyplot as plt


# linear svm

acc1 = np.array([0.569565217391, 0.647826086957, 0.886956521739,
    0.908695652174, 0.913043478261])
fpr1 = np.array([0.0931876606684, 0.140102827763, 0.174807197943,
    0.143958868895, 0.115681233933])


# random classifier
rand = np.arange(0, 1.1, 0.2)


# This is the ROC curve

plt.plot(fpr1, acc1, 'r', label='linear svm')
plt.plot(rand, rand, 'k', label='random')
#plt.legend(loc='upper left')
plt.title('Linear SVM model')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.show() 


