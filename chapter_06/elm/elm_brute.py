#
#  file: elm_brute.py
#
#  If first you don't succeed, try, try again...
#
#  RTK, 23-Jun-2022
#  Last update, 23-Jun-2022
#
####################################################

import sys
import pickle
from RE import *

# activation functions
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
    print("elm <nodes> <act> <target> <max> <output> [<kind>]")
    print()
    print("  <nodes>  - number of nodes in the hidden layer (e.g. 100)")
    print("  <act>    - activation function: relu,tanh,sigmoid,cube,absolute,recip")
    print("  <target> - match or exceed this test accuracy (fraction)")
    print("  <max>    - give up after this many trials")
    print("  <output> - output name for model (.pkl)")
    print("  <kind>   - randomness source")
    print()
    exit(0)

nodes = int(sys.argv[1])
act = sys.argv[2].lower()
target = float(sys.argv[3])
miter = int(sys.argv[4])
oname = sys.argv[5]

#  rng accessed globally
if (len(sys.argv) == 7):
    rng = RE(kind=sys.argv[6])
else:
    rng = RE()

#  set up the activation function
acts = {
    "relu": relu,
    "tanh" : tanh,
    "sigmoid" : sigmoid,
    "cube" : cube,
    "absolute": absolute,
    "recip" : recip,
}
activation = acts[act]

# load and preprocess the data
xtrn = np.load("../data/datasets/mnist_train_data.npy")/256.0
yn   = np.load("../data/datasets/mnist_train_labels.npy")
xtst = np.load("../data/datasets/mnist_test_data.npy")/256.0
yt   = np.load("../data/datasets/mnist_test_labels.npy")

ytrn = np.zeros((len(yn),4), dtype="uint8")
for i in range(4):
    j = np.where(yn == i)[0]
    ytrn[j,i] = 1

ytst = np.zeros((len(yt),4), dtype="uint8")
for i in range(4):
    j = np.where(yt == i)[0]
    ytst[j,i] = 1

#  Run up to miter trials
success = False
for k in range(miter):
    model = train(xtrn, ytrn, nodes)
    prob = predict(xtst,model)
    cm,acc = confusion(prob,ytst)
    if (acc >= target):
        success = True
        niter = k+1
        break

if (success):
    print("Success: %0.6f after %d iterations:" % (acc,niter))
    print()
    print(cm)
    pickle.dump(model, open(oname,"wb"))
else:
    print("No suitable model found after %d iterations" % miter)

