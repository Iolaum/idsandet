#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Compare SVM performance in two spaces.

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

## knn sq eucl - knn sq sd eucl - kmc - svm


acc1 = np.array([0.55, 0.53, 0.59, 0.82, 0.89, 0.91, 0.912, 0.91, 0.93, 1])
fpr1 = np.array([0.19, 0.22, 0.30, 0.41, 0.49, 0.61, 0.73, 0.81, 0.904, 1])

acc2 = np.array([0.55, 0.58, 0.72, 0.826, 0.876, 0.9269, 0.930, 0.936, 0.9560, 1])
fpr2 = np.array([0.20, 0.245, 0.32, 0.41, 0.5, 0.64, 0.76, 0.86, 0.909, 1])

per1 = acc1 - fpr1
per2 = acc2 - fpr2

xax = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9, 1.])


plt.plot(xax, per1, 'r', label='frequency space')
plt.plot(xax, per2, 'g', label='two-sequence space')

#plt.tight_layout()
plt.legend(loc='lower left')
plt.title('Performance of one-class SVM classifier')
plt.ylabel('Scorer')
plt.xlabel('training error bound')
#plt.show() 
plt.savefig('../pictures/f03-results.eps')
plt.close()
