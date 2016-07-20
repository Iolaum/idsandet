#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves

import numpy as np
import matplotlib.pyplot as plt


# sq eucl - adduser

acc1 = np.array([0.912087912088, 0.868131868132, 0.681318681319,
    0.571428571429, 0.252747252747, 0.120879120879, 0.0769230769231,
    0.0549450549451, 0.043956043956, 0.032967032967])
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

# sq sn eucl - adduser
acc7 = np.array([0.912087912088, 0.67032967033, 0.56043956044,
    0.120879120879, 0.0549450549451, 0.032967032967, 0.021978021978,
    0.021978021978, 0.021978021978, 0.010989010989])
fpr7 = np.array([0.75571820677, 0.308325709058, 0.189387008234,
    0.102927721866, 0.0574107959744, 0.0315645013724, 0.0180695333943,
    0.0176120768527, 0.0176120768527, 0.0173833485819])

# sq sn eucl - hydra ftp
acc8 = np.array([0.703703703704, 0.58024691358, 0.351851851852,
    0.0987654320988, 0.0493827160494, 0.00617283950617, 0, 0, 0, 0])
fpr8 = np.array([0.75571820677, 0.308325709058, 0.189387008234,
    0.102927721866, 0.0574107959744, 0.0315645013724, 0.0180695333943,
    0.0176120768527, 0.0176120768527, 0.0173833485819])

# sq sn eucl - hydra ssh
acc9 = np.array([0.721590909091, 0.392045454545, 0.267045454545,
    0.0795454545455, 0.0568181818182, 0.0454545454545, 0.0113636363636,
    0.0113636363636, 0, 0])
fpr9 = np.array([0.75571820677, 0.308325709058, 0.189387008234,
    0.102927721866, 0.0574107959744, 0.0315645013724, 0.0180695333943,
    0.0176120768527, 0.0176120768527, 0.0173833485819])

# This is the ROC curve
'''
plt.plot(fpr1, acc1, 'r', label='adduser')
plt.plot(fpr1, acc2, 'g', label='hydra ftp')
plt.plot(fpr1, acc3, 'b', label='hydra ssh')
plt.plot(fpr1, fpr1, 'k', label='random')
plt.legend(loc='upper left')
plt.title('kNN with squared euclidean distance')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.show() 
'''

plt.plot(fpr7, acc7, 'r', label='adduser')
plt.plot(fpr7, acc8, 'g', label='hydra ftp')
plt.plot(fpr7, acc9, 'b', label='hydra ssh')
plt.plot(fpr7, fpr7, 'k', label='random')
plt.legend(loc='upper left')
plt.title('kNN with squared standardised euclidean distance')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.show() 