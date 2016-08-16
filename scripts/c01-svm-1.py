#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implement an SVM classifier


import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

# where to log results
resultsfile = '../data/c01-svm-2-results.txt'
# custom print function to also save runs on text file
def myprint(mytext):
    print(mytext)
    with open(resultsfile, 'a') as ha:
        ha.write(mytext + '\n')

myprint("Starting a linear SVM classifier.")

myprint("Loading training data.")
trdat = np.load('../data/b5_trmatrix.npy')

# attack data options
atdatselector = 1

if atdatselector == 1:
    myprint("Loading adduser attack data.")
    atdat = np.load('../data/b5_at1mat_adduser.npy')

elif atdatselector == 2:
    myprint("Loading hydra ftp attack data.")
    atdat = np.load('../data/b5_at2mat_hyftp.npy')

elif atdatselector == 3:
    myprint("Loading hydra ssh attack data.")
    atdat = np.load('../data/b5_at3mat_hyssh.npy')

elif atdatselector == 4:
    myprint("Loading java meterpreter attack data.")
    atdat = np.load('../data/b5_at4mat_javamet.npy')

elif atdatselector == 5:
    myprint("Loading meterpreter attack data.")
    atdat = np.load('../data/b5_at5mat_meter.npy')

elif atdatselector == 6:
    myprint("Loading web shell attack data.")
    atdat = np.load('../data/b5_at6mat_webshell.npy')


myprint("Loading validation data.")
vadat = np.load('../data/b5_vamatrix.npy')

# split attach set
at1ctr = atdat.shape[0]/2


# use to train
at1 = atdat[0:at1ctr, :]

# use for accuracy
at2 = atdat[at1ctr:, :]

# create classification labels
l1 = np.zeros(trdat.shape[0])
l2 = np.ones(at1.shape[0])


# convatenate to create training set
x = np.concatenate((trdat, at1), axis=0)
y = np.concatenate((l1, l2), axis=0)

# debug
# print x.shape
# print y.shape

# define SVM parameters
# cpar 0.05 - 0.1 - 0.5 - 1 - 5 - 10 - 50
cpar = 0.1
model = SVC(C=cpar, kernel='linear', max_iter=5000, verbose=False, class_weight='balanced')

# Influence of C
# http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

myprint("Starting training an SVM model with linear kernel and C={}.".format(cpar))

# model.fit(x[830:930, :], y[830:930])
model.fit(x, y)

# log support vectors
myprint('Number of Support Vectors for normal class: {}'.format(model.n_support_[0]))
myprint('Number of Support Vectors for attack class: {}'.format(model.n_support_[1]))

pre1 = model.predict(at2)
pre1a1 = float(at2.shape[0])
pre1a2 = np.sum(pre1)
myprint("Attack detection accuracy in our SVM model is {}".format(pre1a2/pre1a1))

pre2 = model.predict(vadat)
pre2a1 = float(vadat.shape[0])
pre2a2 = np.sum(pre2)
myprint("Attack detection false positive rate in our SVM model is {}".format(pre2a2/pre2a1))


with open(resultsfile, 'a') as ha:
    ha.write('\n')
