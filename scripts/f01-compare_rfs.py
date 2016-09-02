#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Compare Reduced Frequency space Performance

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

## knn sq eucl - knn sq sd eucl - kmc - svm

# adduser attack
att1 = [ 0.696, 0.745, 0.6893, 0.8303]

# hydra ftp attack
att2 = [0.574, 0.593, 0.6428, 0.6978]

# hydra ssh attack
att3 = [0.518, 0.549, 0.4690, 0.7801]

# java meterpreter attack
att4 = [0.689, 0.727, 0.6858, 0.8628]

# meterpreter attack
att5 = [0.705, 0.710, 0.7475, 0.9105]

# web shell attack
att6 = [0.697, 0.734, 0.7158, 0.8354]

xax = [1,2,3,4]
index = ['knn sq sd eucl', 'knn sq eucl', 'kmc', 'svm']



plt.plot(xax, att1, 'r', label='adduser')
plt.plot(xax, att2, 'g', label='hydra ftp')
plt.plot(xax, att3, 'b', label='hydra ssh')
plt.plot(xax, att4, 'c', label='java meter')
plt.plot(xax, att5, 'm', label='meterpreter')
plt.plot(xax, att6, 'y', label='web shell')
plt.xticks([1., 2., 3., 4.], tuple(index))
#plt.tight_layout()
plt.legend(loc='upper left')
plt.title('Performance on reduced frequency space')
plt.ylabel('ROC area')
plt.xlabel('Algorithm used')
#plt.show() 
plt.savefig('../pictures/f01-results.eps')
plt.close()
