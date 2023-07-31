#
#  file: forest.py
#
#  An ensemble of decision trees using bagging, ensembling and 
#  random feature selection -- i.e., a random forest
#
#  RTK, 25-Jun-2022
#  Last update:  25-Jun-2022
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
    nf = xtrn.shape[1]
    m = np.argsort(RE().random(nf))[:int(np.sqrt(nf))]
    return xtrn[n][:,m], ytrn[n], m

if (len(sys.argv) == 1):
    print()
    print("forest <N>")
    print()
    print("  <N> - number of decision trees (e.g. 60)")
    print()
    exit(0)

N = int(sys.argv[1])

xtrn = np.load("../data/datasets/bc_train_data.npy")
ytrn = np.load("../data/datasets/bc_train_labels.npy")
xtst = np.load("../data/datasets/bc_test_data.npy")
ytst = np.load("../data/datasets/bc_test_labels.npy")

# train a collection of N decision trees using bagging
trees = []
for i in range(N):
    tr = DecisionTreeClassifier()
    x,y,m = Bootstrap(xtrn,ytrn)
    tr.fit(x,y)
    trees.append((tr,m))

# apply each tree to the test data
preds = []
for i in range(N):
    tr,m = trees[i]
    preds.append(tr.predict(xtst[:,m]))
preds = np.array(preds)

# average over models and round to nearest integer
pred = np.floor(preds.mean(axis=0) + 0.5).astype("uint8")

# calculate the ensemble confusion matrix
cm, acc = Confusion(pred, ytst)
print("Forest with %d decision trees:" % N)
print(cm)
print("overall accuracy %0.4f" % acc)
print()

