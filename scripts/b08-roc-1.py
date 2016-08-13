#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves for k-NN square euclidean distance

import numpy as np
import matplotlib.pyplot as plt


# sq eucl - adduser

acc1 = np.array([0.912087912088, 0.868131868132, 0.681318681319,
    0.571428571429, 0.252747252747, 0.120879120879, 0.0769230769231,
    0.0549450549451, 0.043956043956,  0.032967032967])
fpr1 = np.array([0.775617566331, 0.394098810613, 0.25571820677,
    0.180695333943, 0.121912168344, 0.077538883806, 0.0583257090576,
    0.0365965233303, 0.0292772186642, 0.0267612076853])

# sq eucl - hydra ftp

acc2 = np.array([0.728395061728, 0.635802469136, 0.561728395062,
    0.364197530864, 0.191358024691, 0.0246913580247, 0, 0, 0, 0])
# fpr2 = fpr1

# sq eucl - hydra ssh

acc3 = np.array([0.795454545455, 0.477272727273, 0.409090909091,
    0.25, 0.181818181818, 0.0625, 0.0170454545455, 0.00568181818182,
    0.00568181818182, 0])


# sq eucl - java meterpreter

acc4 = np.array([0.887096774194, 0.798387096774, 0.620967741935,
    0.508064516129, 0.395161290323, 0.258064516129, 0.241935483871,
    0.193548387097, 0.161290322581, 0.0967741935484])

# sq eucl - meterpreter

acc5 = np.array([0.866666666667, 0.8, 0.733333333333, 0.52, 0.24, 0.04,
    0.0133333333333, 0.0133333333333, 0.0133333333333, 0])


# sq eucl - web shell

acc6 = np.array([0.957627118644, 0.85593220339, 0.593220338983,
    0.432203389831, 0.21186440678, 0.0932203389831, 0.0762711864407,
    0.0677966101695, 0.0677966101695, 0.0677966101695])




# This is the ROC curve

plt.plot(fpr1, acc1, 'r', label='adduser')
plt.plot(fpr1, acc2, 'g', label='hydra ftp')
plt.plot(fpr1, acc3, 'b', label='hydra ssh')
plt.plot(fpr1, acc4, 'c', label='java meter')
plt.plot(fpr1, acc5, 'm', label='meterpreter')
plt.plot(fpr1, acc6, 'y', label='web shell')
plt.plot(fpr1, fpr1, 'k', label='random')
plt.legend(loc='lower right')
plt.title('kNN with squared euclidean distance')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
#plt.show() 
plt.savefig('../pictures/b08-roc-1.eps')
plt.close()
