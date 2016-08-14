#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves for k-NN square standardised euclidean distance

import numpy as np
import matplotlib.pyplot as plt


# sq sn eucl - adduser

acc1 = np.array([0.912087912088, 0.67032967033,  0.56043956044,
    0.120879120879, 0.0549450549451, 0.032967032967, 0.021978021978,
    0.021978021978,  0.021978021978, 0.010989010989])
fpr1 = np.array([0.75571820677,  0.308325709058, 0.189387008234,
    0.102927721866, 0.0574107959744, 0.0315645013724, 0.0180695333943,
    0.0176120768527, 0.0176120768527, 0.0173833485819])

# sq sn eucl - hydra ftp

acc2 = np.array([0.703703703704, 0.58024691358, 0.351851851852,
    0.0987654320988, 0.0493827160494, 0.00617283950617, 0, 0, 0, 0])
# fpr2 = fpr1

# sq sn eucl - hydra ssh

acc3 = np.array([0.721590909091, 0.392045454545, 0.267045454545,
    0.0795454545455, 0.0568181818182, 0.0454545454545, 0.0113636363636,
    0.0113636363636, 0, 0])


# sq sn eucl - java meterpreter

acc4 = np.array([0.846774193548, 0.645161290323, 0.564516129032,
    0.258064516129, 0.25, 0.225806451613, 0.225806451613, 0.217741935484,
    0.209677419355, 0.193548387097])

# sq sn eucl - meterpreter

acc5 = np.array([0.853333333333, 0.76, 0.613333333333, 0.0933333333333,
    0.0666666666667, 0.0266666666667, 0, 0, 0, 0])


# sq sn eucl - web shell

acc6 = np.array([0.957627118644, 0.661016949153, 0.457627118644,
    0.118644067797, 0.0508474576271, 0.0169491525424, 0, 0, 0, 0])




# This is the ROC curve

plt.plot(fpr1, acc1, 'r', label='adduser')
plt.plot(fpr1, acc2, 'g', label='hydra ftp')
plt.plot(fpr1, acc3, 'b', label='hydra ssh')
plt.plot(fpr1, acc4, 'c', label='java meter')
plt.plot(fpr1, acc5, 'm', label='meterpreter')
plt.plot(fpr1, acc6, 'y', label='web shell')
plt.plot(fpr1, fpr1, 'k', label='random')
plt.legend(loc='lower right')
plt.title('kNN with squared standardised euclidean distance')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
#plt.show() 
plt.savefig('../pictures/b08-roc-2.jpg')
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
