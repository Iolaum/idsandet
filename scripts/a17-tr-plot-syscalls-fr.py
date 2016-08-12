#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt

# debug
# xx = [1,2]
# xx = append0(3, xx)
#
# print xx
# exit()


print("Started processing training data!")


trmatrix = np.load("../data/a16_trmatrix.npy")
atmatrix = np.load("../data/a16_atmatrix.npy")
vamatrix = np.load("../data/a16_vamatrix.npy")

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
    #plt.title('Counting system calls.')
    plt.axis([xx[0] - mywidth, xx[-1] + 2 * mywidth, 0.1, maxval])
    bar1 = plt.bar(np.array(xx) - mywidth, yy, color='green', width=mywidth, log=True)
    bar2 = plt.bar(xx, aa, color='red', width=mywidth, log=True)
    bar3 = plt.bar(np.array(xx) + mywidth, va, color='blue', width=mywidth, log=True)
    # plt.show()
    plt.legend((bar1, bar2, bar3), ('training', 'attack', 'validation'))
    plt.savefig('../pictures/a17-syscalls-{}.eps'.format(int(it/25)))
    print("Figure-{} saved.".format(int(it/25)))
    plt.close()
    

# load non-zero sys calls
with open('../data/a14-sys-set.p', 'rb') as ha:
    syset = pickle.load(ha)

syslist = list(syset)
syslist.sort
del syset

# debug
# print len(syslist)
# exit()
# 175

# syscall number
syscalls = range(1, 326)


xx = []
yy = []
aa = []
va = []


# count absolute frequency of system calls
for it in range(len(syslist)):
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
            xx.append(it)
            yy.append(trmatrix[0, it])
            aa.append(atmatrix[0, it])
            va.append(vamatrix[0, it])


    else:
    # just append to create graph!
        xx.append(it)
        yy.append(trmatrix[0, it])
        aa.append(atmatrix[0, it])
        va.append(vamatrix[0, it])

# print last graph
gprint(175, xx, yy, aa, va)

print("System call statistics successfully plotted!.") 

