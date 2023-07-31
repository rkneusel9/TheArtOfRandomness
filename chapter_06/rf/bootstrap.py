import numpy as np
from RE import *

def bootstrap(x):
    n = RE(mode="int", low=0, high=len(x)).random(len(x))
    return x[n]

x = np.load("iris_train_data.npy")[:,0]

means = [x.mean()]

for i in range(10000):
    y = bootstrap(x)
    means.append(y.mean())
means = np.array(means)

L = np.quantile(means, 0.025)
U = np.quantile(means, 0.975)

print()
print("mean from single measurement %0.4f" % x.mean())
print("population mean 95%% confidence interval [%0.4f, %0.4f]" % (L,U))
print()

