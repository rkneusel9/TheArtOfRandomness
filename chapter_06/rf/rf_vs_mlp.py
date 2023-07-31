import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def Confusion(p,y):
    nc = y.max() + 1
    cm = np.zeros((nc,nc), dtype="uint16")
    for i in range(len(p)):
        cm[y[i],p[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm,acc

def Run(xtrn,ytrn, xtst, ytst, mlp=False):
    acc = []
    for i in range(16):
        if (mlp):
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000)
        else:
            clf = RandomForestClassifier(n_estimators=60)
        clf.fit(xtrn, ytrn)
        p = clf.predict(xtst)
        _, a = Confusion(p, ytst)
        acc.append(a)
    return np.array(acc)

xtrn = np.load("../data/datasets/mnist_14x14_xtrn.npy")
ytrn = np.load("../data/datasets/mnist_14x14_ytrn.npy")
xtst = np.load("../data/datasets/mnist_14x14_xtst.npy")
ytst = np.load("../data/datasets/mnist_14x14_ytst.npy")

#  Scale each feature unevently
factors = 10000*(np.random.random(size=xtrn.shape[1]) - 0.5)
xtrn = factors*xtrn
xtst = factors*xtst

#  No scaling
mlp0 = Run(xtrn,ytrn, xtst,ytst, mlp=True)
rf0 = Run(xtrn,ytrn, xtst,ytst, mlp=False)

xtrn = np.load("../data/datasets/mnist_14x14_xtrn.npy")/256.0
xtst = np.load("../data/datasets/mnist_14x14_xtst.npy")/256.0

#  Scaled
mlp1 = Run(xtrn,ytrn, xtst,ytst, mlp=True)
rf1 = Run(xtrn,ytrn, xtst,ytst, mlp=False)

np.save("rf_vs_mlp_mlp0.npy", mlp0)
np.save("rf_vs_mlp_mlp1.npy", mlp1)
np.save("rf_vs_mlp_rf0.npy", rf0)
np.save("rf_vs_mlp_rf1.npy", rf1)

