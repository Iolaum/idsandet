#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Compare SVM performance in two spaces.

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

## knn sq eucl - knn sq sd eucl - kmc - svm


acc1 = np.array([0.62, 0.64, 0.66, 0.68, 0.76, 0.81, 0.91, 0.91, 0.92])
fpr1 = np.array([0.17, 0.17, 0.16, 0.16, 0.19, 0.19, 0.18, 0.18, 0.18])

acc2 = np.array([0.93, 0.93, 0.92, 0.91, 0.91, 0.88, 0.86, 0.84, 0.81])
fpr2 = np.array([0.068, 0.062, 0.054, 0.049, 0.047, 0.046, 0.043, 0.041, 0.038])

per1 = acc1 - fpr1
per2 = acc2 - fpr2

#print per1
#print per2


xax = np.array([0.125, 0.25, 0.5, 1., 2., 4., 8., 16., 32.])


plt.semilogx(xax, per1, 'r', label='frequency space')
plt.semilogx(xax, per2, 'g', label='two-sequence space')

#plt.tight_layout()
plt.legend(loc='lower right')
plt.title('Performance of SVM classifier')
plt.ylabel('Scorer')
plt.xlabel('Regularisation parameter')
#plt.show() 
plt.savefig('../pictures/f02-results.eps')
plt.close()
