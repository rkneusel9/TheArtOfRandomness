#
#  file: elm_swarm.py
#
#  Extreme learning machine swarm search
#
#  RTK, 22-Jun-2022
#  Last update, 22-Jun-2022
#
####################################################

import sys
import time
import pickle
import numpy as np

sys.path.append("include")

from GA import *
from DE import *
from Jaya import *
from GWO import *
from RO import *
from PSO import *
from Bounds import *
from LinearInertia import *
from RandomInitializer import *
from RE import *

#  global randomness source
rng = RE()

# activation functions -- assign to "activation"
def relu(x):
    return np.maximum(x,0)

def cube(x):
    return x**3

def sigmoid(x):
    return 1 / (1 + np.exp(-0.01*x))
    
def tanh(x):
    return np.tanh(0.01*x)

def absolute(x):
    return np.abs(x)

def recip(x):
    return 1/x

# use tanh
activation = tanh
        
def train(xtrn, ytrn, w0, b0):
    """Train an elm"""    
    inp = xtrn.shape[1]
    z = activation(np.dot(xtrn,w0) + b0)
    zinv = np.linalg.pinv(z)
    w1 = np.dot(zinv, ytrn)
    return (w0,b0,w1)

def predict(xtst, model):
    w0,bias,w1 = model
    z = activation(np.dot(xtst,w0) + bias)
    return np.dot(z,w1)

def confusion(prob,ytst):
    nc = prob.shape[1]
    cm = np.zeros((nc,nc), dtype="uint16")
    for i in range(len(prob)):
        n = np.argmax(ytst[i])
        m = np.argmax(prob[i])
        cm[n,m] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm,acc

class Objective:
    def __init__(self, xtrn, ytrn, xtst, ytst, hidden):
        self.xtrn = xtrn;  self.ytrn = ytrn
        self.xtst = xtst;  self.ytst = ytst
        self.hidden = hidden
        self.fcount = 0

    def Evaluate(self, p):
        self.fcount += 1
        n = self.xtrn.shape[1]*self.hidden
        w0 = p[:n].reshape((self.xtrn.shape[1],self.hidden))
        b0 = p[n:]
        model = train(self.xtrn, self.ytrn, w0, b0)
        pred = predict(self.xtst, model)
        cm,acc = confusion(pred, self.ytst)
        return 1.0 - acc


def dist(swarm):
    """Return the mean distance between swarm particles"""

    def dd(a,b):
        return np.sqrt(((a-b)**2).sum())

    p = swarm.pos
    d = []
    for i in range(len(p)):
        for j in range(len(p)):
            if (i==j):
                continue
            d.append(dd(p[i],p[j]))
    return np.array(d).mean()


#
#  main
#

if (len(sys.argv) == 1):
    print()
    print("elm_swarm <hidden> <act> <npart> <niter> <alg> <output>")
    print()
    print("  <hidden> - number hidden layer nodes")
    print("  <act>    - tanh, sigmoid, relu, cube, absolute, recip")
    print("  <npart>  - number of particles")
    print("  <niter>  - number of iterations")
    print("  <alg>    - algorithm type")
    print("  <output> - output file (.pkl)")
    print()
    exit(0)

hidden  = int(sys.argv[1])
act = sys.argv[2].lower()
npart = int(sys.argv[3])
niter = int(sys.argv[4])
alg = sys.argv[5].upper()
oname = sys.argv[6]

# load and preprocess the data
xtrn = np.load("../data/datasets/mnist_train_data.npy")/256.0
yn   = np.load("../data/datasets/mnist_train_labels.npy")
xtst = np.load("../data/datasets/mnist_test_data.npy")/256.0
yt   = np.load("../data/datasets/mnist_test_labels.npy")

# make one-hot encoded
nc = yn.max() + 1
ytrn = np.zeros((len(yn),nc), dtype="uint8")
for i in range(nc):
    j = np.where(yn == i)[0]
    ytrn[j,i] = 1

ytst = np.zeros((len(yt),nc), dtype="uint8")
for i in range(nc):
    j = np.where(yt == i)[0]
    ytst[j,i] = 1

#  Activation function
acts = {
    "tanh" : tanh,
    "relu" : relu,
    "sigmoid" : sigmoid,
    "cube" : cube,
    "absolute" : absolute,
    "recip" : recip
}

activation = acts[act]

#  Setup
ndim = xtrn.shape[1]*hidden + hidden
lower = xtrn.min()
upper = xtrn.max()
b = Bounds([lower]*ndim, [upper]*ndim, enforce="resample", rng=rng)
i = RandomInitializer(npart, ndim, bounds=b, rng=rng)
obj = Objective(xtrn, ytrn, xtst, ytst, hidden)
tol = 0

if (alg == "BARE"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, bare=True, rng=rng)
elif (alg == "PSO"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng, inertia=LinearInertia())
elif (alg == "JAYA"):
    swarm = Jaya(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
elif (alg == "GA"):
    swarm = GA(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
elif (alg == "DE"):
    swarm = DE(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
elif (alg == "GWO"):
    swarm = GWO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
elif (alg == "RO"):
    swarm = RO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
else:
    raise ValueError("Unknown algorithm: %s" % alg)

swarm.Initialize()
k = 0
while (not swarm.Done()):
    swarm.Step()
    res = swarm.Results()
    print("%3d: %0.5f (mean swarm distance %0.9f)" % (k, 1.0 - res["gbest"][-1], dist(swarm)))
    k += 1

res = swarm.Results()
updates = len(res["gbest"])
n = xtrn.shape[1]*hidden
w0 = res["gpos"][-1][:n].reshape((xtrn.shape[1],hidden))
b0 = res["gpos"][-1][n:]
model = train(xtrn, ytrn, w0, b0)
pred = predict(xtst, model)
cm,acc = confusion(pred, ytst)
print(cm)
print("final accuracy = %0.6f (%s-%s-%d, %d:%d, %d models evaluated, %d best updates)" % (acc,alg,act,hidden,npart,niter, obj.fcount, updates))
pickle.dump(model, open(oname,"wb"))

