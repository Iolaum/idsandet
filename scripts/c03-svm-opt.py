#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Print classification results to select best performing parameter C for each attack

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



clper = []
for it in range(len(fpr1)):
    clper.append(acc1[it] - fpr1[it])
print("Adduser attack")
print clper
print max(clper)

clper = []
for it in range(len(fpr2)):
    clper.append(acc2[it] - fpr2[it])
print("hydra ftp")
print clper
print max(clper)

clper = []
for it in range(len(fpr3)):
    clper.append(acc3[it] - fpr3[it])
print("hydra ssh")
print clper
print max(clper)

clper = []
for it in range(len(fpr4)):
    clper.append(acc4[it] - fpr4[it])
print("java meter")
print clper
print max(clper)

clper = []
for it in range(len(fpr5)):
    clper.append(acc5[it] - fpr5[it])
print("meterpreter")
print clper
print max(clper)

clper = []
for it in range(len(fpr6)):
    clper.append(acc6[it] - fpr6[it])
print("webshell")
print clper
print max(clper)



'''
results:
C = 0.05, 0.1, 0.5, 1, 5, 10, 50

Adduser attack - 10
[0.41356060304699993, 0.41584788575599996, 0.40944349417299997, 0.47718684116299998, 0.48061776522599997, 0.48199013485099995, 0.46162337403999998]
0.481990134851
hydra ftp - 0.05
[0.39567449425699996, 0.383568838739, 0.34853952763399998, 0.22833858561199993, 0.21185885489099998, 0.21414613759899992, 0.221922898807]
0.395674494257
hydra ssh - 0.5
[0.19862970972300004, 0.27491058803900004, 0.48147301006400001, 0.2310571404809999, 0.22007818348199992, 0.22305165100299995, 0.23853239624]
0.481473010064
java meter - 0.1
[0.60973792167200003, 0.63615234778500007, 0.50677330814799992, 0.52816309063599998, 0.56064988342299993, 0.57174689372299992, 0.56317327273299989]
0.636152347785
meterpreter - 10
[0.64905860259100001, 0.7021235614200001, 0.70555448548200006, 0.711043963981, 0.70578321375300002, 0.745148552992, 0.735541965618]
0.745148552992
webshell - 1
[0.42935785507200003, 0.42412036534500003, 0.476340192597, 0.48503186688799999, 0.47018778978699999, 0.45852264797499998, 0.47204087645600001]
0.485031866888

'''
