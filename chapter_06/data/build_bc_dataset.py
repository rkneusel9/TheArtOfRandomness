#
#  file:  build_bc_dataset.py
#
#  RTK, 08-Jun-2022
#  Last update:  08-Jun-2022
#
################################################################

import numpy as np
from sklearn import decomposition

def generateData(pca, x, start):
    original = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(x)
    for i in range(start, ncomp):
        pca.components_[i,:] += np.random.normal(scale=0.1, size=ncomp)
    b = pca.inverse_transform(a)
    pca.components_ = original.copy()
    return b


#  Main
np.random.seed(8675309)

x = np.load("raw/bc_data.npy")
y = np.load("raw/bc_labels.npy")

x = (x - x.mean(axis=0)) / x.std(ddof=1,axis=0)

i = np.argsort(np.random.random(len(y)))
x = x[i]
y = y[i]

n = int(0.7*len(y))
xtrn = x[:n]
ytrn = y[:n]
xtst = x[n:]
ytst = y[n:]

pca = decomposition.PCA(n_components=xtrn.shape[1])
pca.fit(x)
print(pca.explained_variance_ratio_)
start = 24

nsets = 10
nsamp = xtrn.shape[0]
newx = np.zeros((nsets*nsamp, xtrn.shape[1]))
newy = np.zeros(nsets*nsamp, dtype="uint8")

for i in range(nsets):
    if (i == 0):
        newx[0:nsamp,:] = xtrn
        newy[0:nsamp] = ytrn
    else:
        newx[(i*nsamp):(i*nsamp+nsamp),:] = generateData(pca, xtrn, start)
        newy[(i*nsamp):(i*nsamp+nsamp)] = ytrn

i = np.argsort(np.random.random(nsets*nsamp))
newx = newx[i]
newy = newy[i]

np.save("datasets/bc_train_data.npy", newx)
np.save("datasets/bc_train_labels.npy", newy)
np.save("datasets/bc_test_data.npy", xtst)
np.save("datasets/bc_test_labels.npy", ytst)

