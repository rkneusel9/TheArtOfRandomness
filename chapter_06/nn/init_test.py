#
#  file:  init_test.py
#
#  Test different network initialization strategies
#
#  RTK, 09-Jun-2022
#  Last update:  20-Jun-2022
#
################################################################

import numpy as np
from Classifier import *
from RE import *
from scipy.stats import ttest_ind, mannwhitneyu

def Confusion(y,p):
    cm = np.zeros((4,4), dtype="uint16")
    for i in range(len(p)):
        cm[y[i],p[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

def Run(init_scheme, xtrn,ytrn,xtst,ytst):
    clf = Classifier(hidden_layer_sizes=(100,50), max_iter=4000)
    clf.init_scheme = init_scheme
    clf.rng = RE()
    clf.fit(xtrn,ytrn)
    pred = clf.predict(xtst)
    _,acc = Confusion(pred,ytst)
    return acc
    

xtrn = np.load("../data/datasets/mnist_train_data.npy")/256.0
ytrn = np.load("../data/datasets/mnist_train_labels.npy")
xtst = np.load("../data/datasets/mnist_test_data.npy")/256.0
ytst = np.load("../data/datasets/mnist_test_labels.npy")

N = 12
init0 = []
for i in range(N):
    init0.append(Run(0, xtrn,ytrn,xtst,ytst))
init0 = np.array(init0)

init1 = []
for i in range(N):
    init1.append(Run(1, xtrn,ytrn,xtst,ytst))
init1 = np.array(init1)

init2 = []
for i in range(N):
    init2.append(Run(2, xtrn,ytrn,xtst,ytst))
init2 = np.array(init2)

init3 = []
for i in range(N):
    init3.append(Run(3, xtrn,ytrn,xtst,ytst))
init3 = np.array(init3)

m0,s0 = init0.mean(), init0.std(ddof=1)/np.sqrt(N)
m1,s1 = init1.mean(), init1.std(ddof=1)/np.sqrt(N)
m2,s2 = init2.mean(), init2.std(ddof=1)/np.sqrt(N)
m3,s3 = init3.mean(), init3.std(ddof=1)/np.sqrt(N)

print()
print("init0: %0.5f +/- %0.5f" % (m0,s0))
print("init1: %0.5f +/- %0.5f" % (m1,s1))
print("init2: %0.5f +/- %0.5f" % (m2,s2))
print("init3: %0.5f +/- %0.5f" % (m3,s3))
print()

#  Always compare He (init 3) against the others in expected order:
_,p = ttest_ind(init3,init0)
_,u = mannwhitneyu(init3,init0)
print("init3 vs init0: p=%0.8f, u=%0.8f" % (p,u))
_,p = ttest_ind(init3,init2)
_,u = mannwhitneyu(init3,init2)
print("init3 vs init2: p=%0.8f, u=%0.8f" % (p,u))
_,p = ttest_ind(init3,init1)
_,u = mannwhitneyu(init3,init1)
print("init3 vs init1: p=%0.8f, u=%0.8f" % (p,u))

