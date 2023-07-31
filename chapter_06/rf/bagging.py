#
#  file: bagging.py
#
#  An ensemble of decision trees using bagging.
#
#  RTK, 24-Jun-2022
#  Last update:  24-Jun-2022
#
################################################################

import sys
import numpy as np
from RE import *
from sklearn.tree import DecisionTreeClassifier

def Confusion(pred, ytst):
    """Confusion matrix and overall accuracy"""
    n = ytst.max() + 1
    cm = np.zeros((n,n), dtype="uint16")
    for i in range(len(pred)):
        cm[ytst[i],pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc

def Bootstrap(xtrn, ytrn):
    n = RE(mode="int", low=0, high=len(xtrn)).random(len(xtrn))
    return xtrn[n], ytrn[n]

if (len(sys.argv) == 1):
    print()
    print("bagging <N> 0|1")
    print()
    print("  <N> - number of decision trees (e.g. 60)")
    print("  0|1 - 0=no bagging, 1=bagging")
    print()
    exit(0)

N = int(sys.argv[1])
bag = int(sys.argv[2])

xtrn = np.load("../data/datasets/bc_train_data.npy")
ytrn = np.load("../data/datasets/bc_train_labels.npy")
xtst = np.load("../data/datasets/bc_test_data.npy")
ytst = np.load("../data/datasets/bc_test_labels.npy")

# train a collection of N decision trees using bagging
trees = []
for i in range(N):
    tr = DecisionTreeClassifier()
    if (bag):
        x,y = Bootstrap(xtrn,ytrn)
        tr.fit(x,y)
    else:
        tr.fit(xtrn,ytrn)
    trees.append(tr)

# apply each tree to the test data
preds = []
for i in range(N):
    preds.append(trees[i].predict(xtst))
preds = np.array(preds)

# average over models and round to nearest integer
pred = np.floor(preds.mean(axis=0) + 0.5).astype("uint8")

# calculate the ensemble confusion matrix
cm, acc = Confusion(pred, ytst)
print("Bagging with %d decision trees:" % N)
print(cm)
print("overall accuracy %0.4f" % acc)
print()
_, a0 = Confusion(preds[0], ytst)
_, a1 = Confusion(preds[1], ytst)
_, a2 = Confusion(preds[2], ytst)
_, a3 = Confusion(preds[3], ytst)
_, a4 = Confusion(preds[4], ytst)
_, a5 = Confusion(preds[5], ytst)
print("first six accuracies: %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f" % (a0,a1,a2,a3,a4,a5))
print()

