#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create ROC curves

import numpy as np
import matplotlib.pyplot as plt


# linear svm - adduser

acc1 = np.array([0.54347826087,  0.54347826087,  0.54347826087,
    0.630434782609, 0.630434782609, 0.630434782609, 0.608695652174])
fpr1 = np.array([0.129917657823, 0.127630375114, 0.134034766697,
    0.153247941446, 0.149817017383, 0.148444647758, 0.147072278134])


# linear svm - hydra ftp

acc2 = np.array([0.555555555556, 0.518518518519, 0.592592592593,
    0.851851851852, 0.876543209877, 0.876543209877, 0.876543209877])
fpr2 = np.array([0.159881061299, 0.13494967978,  0.244053064959,
    0.62351326624,  0.664684354986, 0.662397072278, 0.65462031107])


# linear svm - hydra ssh

acc3 = np.array([0.306818181818, 0.431818181818, 0.75, 
    0.886363636364, 0.886363636364, 0.886363636364, 0.897727272727])
fpr3 = np.array([0.108188472095, 0.156907593779, 0.268526989936,
    0.655306495883, 0.666285452882, 0.663311985361, 0.659194876487])

# linear svm - java meter

acc4 = np.array([0.774193548387, 0.758064516129, 0.612903225806, 0.629032258065,
    0.661290322581, 0.677419354839, 0.693548387097])
fpr4 = np.array([0.164455626715, 0.121912168344, 0.106129917658, 0.100869167429,
    0.100640439158, 0.105672461116, 0.130375114364])

# linear svm - meterpreter

acc5 = np.array([0.842105263158, 0.842105263158, 0.842105263158, 0.842105263158,
    0.842105263158, 0.894736842105, 0.894736842105])
fpr5 = np.array([0.193046660567, 0.139981701738, 0.136550777676, 0.131061299177,
    0.136322049405, 0.149588289113, 0.159194876487])

# linear svm - webshell

acc6 = np.array([0.525423728814, 0.542372881356, 0.593220338983, 0.593220338983,
    0.610169491525, 0.610169491525, 0.627118644068])
fpr6 = np.array([0.096065873742, 0.118252516011, 0.116880146386, 0.108188472095,
    0.139981701738, 0.15164684355,  0.155077767612])


# random classifier
rand = np.arange(0, 1.01, 0.2)

# This is the ROC curve
plt.scatter(fpr1, acc1, c='r', label='adduser', marker = "D")
plt.scatter(fpr2, acc2, c='g', label='hydra ftp', marker = "D")
plt.scatter(fpr3, acc3, c='b', label='hydra ssh', marker = "D")
plt.scatter(fpr4, acc4, c='c', label='java meter', marker = "D")
plt.scatter(fpr5, acc5, c='m', label='meterpreter', marker = "D")
plt.scatter(fpr6, acc6, c='y', label='web shell', marker = "D")
plt.plot(rand, rand, 'k', label='random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('linear SVM performance')
plt.ylabel('Accuracy')
plt.xlabel('False Positive')
plt.savefig('../pictures/c02-svm-plot.eps')
