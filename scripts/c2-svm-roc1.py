#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves

import numpy as np
import matplotlib.pyplot as plt


# linear svm - adduser

acc1 = np.array([1.0, 0.54347826087, 0.630434782609, 0.630434782609, 0.608695652174])
fpr1 = np.array([1.0, 0.127630375114, 0.153247941446, 0.148444647758, 0.14592863678])

acc1 = np.array([1.0, 0.630434782609, 0.630434782609, 0.608695652174, 0.54347826087])
fpr1 = np.array([1.0, 0.153247941446, 0.148444647758, 0.14592863678, 0.127630375114])

# linear svm - hydra ftp

acc2 = np.array([0.0, 0.518518518519, 0.851851851852, 0.876543209877, 0.876543209877])
fpr2 = np.array([0.0, 0.13494967978, 0.62351326624, 0.662397072278, 0.657593778591])


# linear svm - hydra ssh

acc3 = np.array([1.0, 0.886363636364, 0.886363636364, 0.886363636364, 0.431818181818])
fpr3 = np.array([1.0, 0.655306495883, 0.663311985361, 0.658508691674, 0.156907593779])

# random classifier
rand = np.arange(0, 1.1, 0.2)

# This is the ROC curve

plt.plot(fpr1, acc1, 'r', label='adduser-alt')
plt.plot(fpr2, acc2, 'g', label='hydra ftp')
plt.plot(fpr3, acc3, 'b', label='hydra ssh')
plt.plot(rand, rand, 'k', label='random')
plt.legend(loc='upper left')
plt.title('linear SVM ROC curve')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.show() 
