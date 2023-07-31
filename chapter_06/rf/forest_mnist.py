#
#  file: forest_mnist.py
#
#  RTK, 25-Jun-2022
#  Last update:  25-Jun-2022
#
################################################################

import numpy as np
import matplotlib.pylab as plt
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

xtrn = np.load("../data/datasets/mnist_train_data.npy")/256.0
ytrn = np.load("../data/datasets/mnist_train_labels.npy")
xtst = np.load("../data/datasets/mnist_test_data.npy")/256.0
ytst = np.load("../data/datasets/mnist_test_labels.npy")

M = 20
ntrees = [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,500]
accuracy = np.zeros((len(ntrees),M))

for j,N in enumerate(ntrees):
    acc = []
    for k in range(M):
        trees = []
        for i in range(N):
            tr = DecisionTreeClassifier()
            x,y,m = Bootstrap(xtrn,ytrn)
            tr.fit(x,y)
            trees.append((tr,m))

        preds = []
        for i in range(N):
            tr,m = trees[i]
            preds.append(tr.predict(xtst[:,m]))
        preds = np.array(preds)

        pred = []
        for i in range(preds.shape[1]):
            pred.append(np.argmax(np.bincount(preds[:,i])))
        pred = np.array(pred)

        cm, a = Confusion(pred, ytst)
        acc.append(a)

    accuracy[j,:] = np.array(acc)

y = accuracy.mean(axis=1)
e = accuracy.std(ddof=1,axis=1) / np.sqrt(M)
plt.errorbar(ntrees, y, e, marker='o', fillstyle='none', color='k', linewidth=0.5)
plt.xlabel("Forest size")
plt.ylabel("Mean accuracy ($n=%d$)" % (M,))
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("forest_mnist_plot.png", dpi=300)
plt.savefig("forest_mnist_plot.eps", dpi=300)
plt.close()

