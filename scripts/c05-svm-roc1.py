#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves

import pickle
import numpy as np
import matplotlib.pyplot as plt

datafile = '../data/c04-svm-1-'



# linear svm - adduser

atdatselector = 1
savefile = datafile +'{}-acc.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmacc1 = pickle.load(ha)

savefile = datafile +'{}-fpr.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmfpr1 = pickle.load(ha)


# linear svm - hydra ftp

atdatselector = 2
savefile = datafile +'{}-acc.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmacc2 = pickle.load(ha)

savefile = datafile +'{}-fpr.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmfpr2 = pickle.load(ha)
# linear svm - hydra ssh

atdatselector = 3
savefile = datafile +'{}-acc.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmacc3 = pickle.load(ha)

savefile = datafile +'{}-fpr.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmfpr3 = pickle.load(ha)

# linear svm - java meter

atdatselector = 4
savefile = datafile +'{}-acc.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmacc4 = pickle.load(ha)

savefile = datafile +'{}-fpr.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmfpr4 = pickle.load(ha)


# linear svm - meterpreter

atdatselector = 5
savefile = datafile +'{}-acc.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmacc5 = pickle.load(ha)

savefile = datafile +'{}-fpr.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmfpr5 = pickle.load(ha)

# linear svm - webshell

atdatselector = 6
savefile = datafile +'{}-acc.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmacc6 = pickle.load(ha)

savefile = datafile +'{}-fpr.p'.format(atdatselector)
with open(savefile, 'rb') as ha:
    svmfpr6 = pickle.load(ha)



# random classifier
rand = np.arange(0, 1.01, 0.2)

# This is the ROC curve
#plt.scatter(fpr1, acc1, c='r', label='adduser', marker = "D")
#plt.scatter(fpr2, acc2, c='g', label='hydra ftp', marker = "D")
#plt.scatter(fpr3, acc3, c='b', label='hydra ssh', marker = "D")
#plt.scatter(fpr4, acc4, c='c', label='java meter', marker = "D")
#plt.scatter(fpr5, acc5, c='m', label='meterpreter', marker = "D")
#plt.scatter(fpr6, acc6, c='y', label='web shell', marker = "D")
plt.plot(svmfpr1, svmacc1, 'r', label='adduser')
plt.plot(svmfpr2, svmacc2, 'g', label='hydra ftp')
plt.plot(svmfpr3, svmacc3, 'b', label='hydra ssh')
plt.plot(svmfpr4, svmacc4, 'c', label='java meter')
plt.plot(svmfpr5, svmacc5, 'm', label='meterprete')
plt.plot(svmfpr6, svmacc6, 'y', label='web shell')
plt.plot(rand, rand, 'k', label='random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("ROC for linear SVM's.")
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.savefig('../pictures/c05-svm-roc.eps')
plt.close()
#plt.show()


# calculate area under the ROC curve
area1 = np.trapz(svmacc1, x=svmfpr1)
area2 = np.trapz(svmacc2, x=svmfpr2)
area3 = np.trapz(svmacc3, x=svmfpr3)
area4 = np.trapz(svmacc4, x=svmfpr4)
area5 = np.trapz(svmacc5, x=svmfpr5)
area6 = np.trapz(svmacc6, x=svmfpr6)


print("Area under the curve for adduser attack          is: {}".format(area1))
print("Area under the curve for hydra ftp attack        is: {}".format(area2))
print("Area under the curve for hydra ssh attack        is: {}".format(area3))
print("Area under the curve for java meterpreter attack is: {}".format(area4))
print("Area under the curve for meterpreter attack      is: {}".format(area5))
print("Area under the curve for web shell attack        is: {}".format(area6))

'''
Area under the curve for adduser attack          is: -0.830283623056
Area under the curve for hydra ftp attack        is: -0.697801949556
Area under the curve for hydra ssh attack        is: -0.780077767612
Area under the curve for java meterpreter attack is: -0.862812841248
Area under the curve for meterpreter attack      is: -0.910564236529
Area under the curve for web shell attack        is: -0.835383488145
'''
