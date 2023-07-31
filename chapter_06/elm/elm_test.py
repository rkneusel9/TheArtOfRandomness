#
#  file: elm_test.py
#
#  Extreme learning machine example
#
#  RTK, 13-Jun-2022
#  Last update, 17-Jun-2022
#
####################################################

import sys
import numpy as np
import matplotlib.pylab as plt
from RE import *

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

def identity(x):
    return x

# set to the desired activation function
activation = relu
        
def train(xtrn, ytrn, hidden=100):
    """Train an elm"""    
    inp = xtrn.shape[1]
    m = xtrn.min()
    d = xtrn.max() - m
    w0 = d*rng.random(inp*hidden).reshape((inp,hidden)) + m
    b0 = d*rng.random(hidden) + m
    z = activation(np.dot(xtrn,w0) + b0)
    zinv = np.linalg.pinv(z)
    w1 = np.dot(zinv, ytrn)
    return (w0,b0,w1)

def predict(xtst, model):
    w0,bias,w1 = model
    z = activation(np.dot(xtst,w0) + bias)
    return np.dot(z,w1)

def confusion(prob,ytst):
    cm = np.zeros((4,4), dtype="uint16")
    for i in range(len(prob)):
        n = np.argmax(ytst[i])
        m = np.argmax(prob[i])
        cm[n,m] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm,acc

#
#  main
#

if (len(sys.argv) == 1):
    print()
    print("elm_test <kind>")
    print()
    print("  <kind>  - randomness source")
    print()
    exit(0)

#  accessed globally -- fixed seed
rng = RE(kind=sys.argv[1])

# load and preprocess the data
xtrn = np.load("../data/datasets/mnist_train_data.npy")/256.0
yn   = np.load("../data/datasets/mnist_train_labels.npy")
xtst = np.load("../data/datasets/mnist_test_data.npy")/256.0
yt   = np.load("../data/datasets/mnist_test_labels.npy")

# make one-hot encoded
ytrn = np.zeros((len(yn),4), dtype="uint8")
for i in range(4):
    j = np.where(yn == i)[0]
    ytrn[j,i] = 1

ytst = np.zeros((len(yt),4), dtype="uint8")
for i in range(4):
    j = np.where(yt == i)[0]
    ytst[j,i] = 1

#  list of activation functions, layer sizes, and trials
acts = [relu, sigmoid, tanh, cube, absolute, recip, identity]
nodes = [10,20,30,40,50,100,150,200,250,300,350,400]
N = 50

#  accuracy grid -- activation, hidden nodes, N trials
acc = np.zeros((len(acts),len(nodes),N))

for i,act in enumerate(acts):
    for j,n in enumerate(nodes):
        for k in range(N):
            activation = act
            model = train(xtrn, ytrn, n)
            prob = predict(xtst, model)
            _,a = confusion(prob, ytst)
            acc[i,j,k] = a

np.save("elm_test_results.npy", acc)

