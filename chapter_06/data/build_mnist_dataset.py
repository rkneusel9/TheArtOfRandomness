#
#  file:  build_mnist_dataset.py
#
#  RTK, 08-Jun-2022
#  Last update:  08-Jun-2022
#
################################################################

import numpy as np
from scipy.ndimage import rotate, zoom

def augment(x):
    im = x.reshape((28,28))

    if (np.random.random() < 0.5):
        angle = -3 + 6*np.random.random()
        im = rotate(im, angle, reshape=False)
    if (np.random.random() < 0.1):
        f = 0.8 + 0.4*np.random.random()
        t = zoom(im, f)
        if (t.shape[0] < 28):
            im = np.zeros((28,28), dtype="uint8")
            c = (28-t.shape[0])//2
            im[c:(c+t.shape[0]),c:(c+t.shape[0])] = t
        if (t.shape[0] > 28):
            c = (t.shape[0]-28)//2
            im = t[c:(c+28),c:(c+28)]

    return im.ravel()


#  Main
np.random.seed(8675309)

x = np.load("raw/mnist_data.npy")
y = np.load("raw/mnist_labels.npy")

i = np.argsort(np.random.random(len(y)))
x = x[i]
y = y[i]

n = int(0.5*len(y))
xtrn = x[:n]
ytrn = y[:n]
xtst = x[n:]
ytst = y[n:]

np.save("datasets/mnist_no_aug_train_data.npy", xtrn)
np.save("datasets/mnist_no_aug_train_labels.npy", ytrn)

newx = []
newy = []

for i in range(len(ytrn)):
    newx.append(xtrn[i])
    newy.append(ytrn[i])
    for j in range(20):
        newx.append(augment(xtrn[i]))
        newy.append(ytrn[i])
xtrn = np.array(newx)
ytrn = np.array(newy)

i = np.argsort(np.random.random(len(ytrn)))
xtrn = xtrn[i]
ytrn = ytrn[i]

np.save("datasets/mnist_train_data.npy", xtrn)
np.save("datasets/mnist_train_labels.npy", ytrn)
np.save("datasets/mnist_test_data.npy", xtst)
np.save("datasets/mnist_test_labels.npy", ytst)

