#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves for k-NN square standardised euclidean distance

import numpy as np
import matplotlib.pyplot as plt


# kmc eucl - adduser

acc1 = np.array([1., 0.813186813187, 0.681318681319,
    0.527472527473, 0.373626373626, 0.131868131868, 0.032967032967,
    0, 0, 0])
fpr1 = np.array([0.995425434584, 0.776989935956, 0.296889295517,
    0.178865507777, 0.0981244281793, 0.0322506861848, 0.0114364135407,
    0.00411710887466, 0.00297346752059, 0.000228728270814])

# kmc eucl - hydra ftp

acc2 = np.array([1, 0.888888888889, 0.567901234568, 0.308641975309,
    0.191358024691, 0.0617283950617, 0, 0, 0, 0])


# kmc eucl- hydra ssh

acc3 = np.array([0.994318181818, 0.630681818182, 0.357954545455,
    0.227272727273, 0.0852272727273, 0, 0, 0, 0, 0])


# kmc eucl - java meterpreter

acc4 = np.array([1., 0.862903225806, 0.612903225806, 0.564516129032,
    0.354838709677, 0.0806451612903, 0, 0, 0, 0])

# kmc eucl - meterpreter

acc5 = np.array([1., 0.96, 0.706666666667, 0.626666666667, 0.213333333333,
    0.16, 0, 0, 0, 0])


# kmc eucl - web shell

acc6 = np.array([1., 0.864406779661, 0.686440677966, 0.559322033898,
    0.423728813559, 0.14406779661, 0, 0, 0, 0])




# This is the ROC curve

plt.plot(fpr1, acc1, 'r', label='adduser')
plt.plot(fpr1, acc2, 'g', label='hydra ftp')
plt.plot(fpr1, acc3, 'b', label='hydra ssh')
plt.plot(fpr1, acc4, 'c', label='java meter')
plt.plot(fpr1, acc5, 'm', label='meterpreter')
plt.plot(fpr1, acc6, 'y', label='web shell')
plt.plot(fpr1, fpr1, 'k', label='random')
plt.legend(loc='lower right')
plt.title('k-means clustering ROC curves')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
#plt.show() 
plt.savefig('../pictures/b08-roc-3kmc.eps')
plt.close()

# calculate area under the ROC curve
area1 = np.trapz(acc1, x=fpr1)
area2 = np.trapz(acc2, x=fpr1)
area3 = np.trapz(acc3, x=fpr1)
area4 = np.trapz(acc4, x=fpr1)
area5 = np.trapz(acc5, x=fpr1)
area6 = np.trapz(acc6, x=fpr1)
# area under the curve for random selection
arear = np.trapz(fpr1, x=fpr1)

print("Area under the curve for adduser attack          is: {}".format(area1))
print("Area under the curve for hydra ftp attack        is: {}".format(area2))
print("Area under the curve for hydra ssh attack        is: {}".format(area3))
print("Area under the curve for java meterpreter attack is: {}".format(area4))
print("Area under the curve for meterpreter attack      is: {}".format(area5))
print("Area under the curve for web shell attack        is: {}".format(area6))
print("Area under the curve for random selection        is: {}".format(arear))

# Normalizing area under roc with 1/(2*arear)

print("Area under the curve for adduser attack          is: {}".format(area1/(2*arear)))
print("Area under the curve for hydra ftp attack        is: {}".format(area2/(2*arear)))
print("Area under the curve for hydra ssh attack        is: {}".format(area3/(2*arear)))
print("Area under the curve for java meterpreter attack is: {}".format(area4/(2*arear)))
print("Area under the curve for meterpreter attack      is: {}".format(area5/(2*arear)))
print("Area under the curve for web shell attack        is: {}".format(area6/(2*arear)))
