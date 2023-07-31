import numpy as np
from sklearn.neural_network import MLPClassifier
from scipy.stats import ttest_ind, mannwhitneyu

x0 = np.load("datasets/mnist_no_aug_train_data.npy")
y0 = np.load("datasets/mnist_no_aug_train_labels.npy")
x1 = np.load("datasets/mnist_train_data.npy")
y1 = np.load("datasets/mnist_train_labels.npy")
xtst = np.load("datasets/mnist_test_data.npy")
ytst = np.load("datasets/mnist_test_labels.npy")

N = 40
a0 = []
a1 = []
for i in range(N):
    c0 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000)
    c1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000)
    c0.fit(x0,y0)
    c1.fit(x1,y1)
    p0 = c0.predict(xtst)
    p1 = c1.predict(xtst)
    cm0 = np.zeros((4,4), dtype="uint16")
    cm1 = np.zeros((4,4), dtype="uint16")
    for i in range(len(ytst)):
        cm0[ytst[i],p0[i]] += 1
        cm1[ytst[i],p1[i]] += 1
    a0.append(np.diag(cm0).sum() / cm0.sum())
    a1.append(np.diag(cm1).sum() / cm1.sum())
a0 = np.array(a0)
a1 = np.array(a1)

print()
print("no augment   = %0.8f +/- %0.8f" % (a0.mean(), a0.std(ddof=1)/np.sqrt(N)))
print("with augment = %0.8f +/- %0.8f" % (a1.mean(), a1.std(ddof=1)/np.sqrt(N)))
print()
_,p = ttest_ind(a0,a1)
_,u = mannwhitneyu(a0,a1)
print("p = %0.6f" % p)
print("u = %0.6f" % u)
print()

