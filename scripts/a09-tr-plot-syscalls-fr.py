#!/usr/bin/env python
# -*- coding: utf-8 -*-

# DEPRECATED!

import pickle
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt


# function to change zero count!
def append0(ap, xx):
    if ap == 0:
        xx.append(0.2)
    else:
        xx.append(ap)
    #print ap  # debug
    return xx


# debug
# xx = [1,2]
# xx = append0(3, xx)
#
# print xx
# exit()


print("Started processing training data!")

with open('../data/a8_trmatrix.p', 'rb') as ha:
    trmatrix = pickle.load(ha)

with open('../data/a8_atmatrix.p', 'rb') as ha:
    atmatrix = pickle.load(ha)

with open('../data/a8_vamatrix.p', 'rb') as ha:
    vamatrix = pickle.load(ha)

maxval1 = np.amax(trmatrix)
maxval2 = np.amax(atmatrix)
maxval3 = np.amax(vamatrix)
maxval = max(maxval1, maxval2, maxval3)

#print maxval
#exit()


# print graph function
# it:
# xx: syscall number
# yy: syscall count

# custom function to create the plots
def gprint(it, xx, yy, aa, va):
    mywidth = 0.3
    plt.xlabel('System Calls')
    plt.ylabel('Count', rotation='horizontal')
    plt.title('Counting system calls.')
    plt.axis([xx[0] - mywidth, xx[-1] + 2 * mywidth, 0.1, maxval])
    plt.bar(np.array(xx) - mywidth, yy, color='green', width=mywidth, log=True)
    plt.bar(xx, aa, color='red', width=mywidth, log=True)
    plt.bar(np.array(xx) + mywidth, va, color='blue', width=mywidth, log=True)
    # plt.show()
    plt.savefig('../pictures/a9-syscalls-{}.jpg'.format(int(it/25)))
    print("Figure-{} saved.".format(int(it/25)))
    plt.close()
    

# syscall number
syscalls = range(1, 326)

#print number of zero calls trmatrix
print ("Number of training set system calls that are not called is: {}".format(np.count_nonzero(trmatrix)))

xx = []
yy = []
aa = []
va = []



# count absolute frequency of system calls
for it in range(325):
    if it % 25 == 0:
        if it == 0:
            pass
        else:
            # print graph?!
            gprint(it, xx, yy, aa, va)

            # debug!
            # exit()

            # reset graph variables.
            xx = []
            yy = []
            aa = []
            va = []

            # start new graph
            # just append to create graph!
            xx.append(syscalls[it])
            yy = append0(trmatrix[0, it], yy)
            aa = append0(atmatrix[0, it], aa)
            va = append0(vamatrix[0, it], va)

    else:
    # just append to create graph!
        xx.append(syscalls[it])
        yy = append0(trmatrix[0, it], yy)
        aa = append0(atmatrix[0, it], aa)
        va = append0(vamatrix[0, it], va)

# print last graph
gprint(325, xx, yy, aa, va)

print("System call statistics successfully plotted!.") 

