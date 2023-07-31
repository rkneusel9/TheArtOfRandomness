#
#  file: elm.py
#
#  Extreme learning machine example
#
#  RTK, 13-Jun-2022
#  Last update, 17-Jun-2022
#
####################################################

import sys
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

def identity(x):
    return x

# set to the desired activation function
activation = tanh
        
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
    nc = ytst.shape[1]
    cm = np.zeros((nc,nc), dtype="uint16")
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
    print("elm <nodes> [<kind>]")
    print()
    print("  <nodes> - number of nodes in the hidden layer (e.g. 100)")
    print("  <kind>  - randomness source")
    print()
    exit(0)

nodes = int(sys.argv[1])

#  rng accessed globally
if (len(sys.argv) == 3):
    rng = RE(kind=sys.argv[2])
else:
    rng = RE()

# load and preprocess the data
xtrn = np.load("../data/datasets/mnist_train_data.npy")/256.0
yn   = np.load("../data/datasets/mnist_train_labels.npy")
xtst = np.load("../data/datasets/mnist_test_data.npy")/256.0
yt   = np.load("../data/datasets/mnist_test_labels.npy")

# 14x14 MNIST dataset w/all digits
#xtrn = np.load("../data/datasets/mnist_14x14_xtrn.npy")/256.0
#yn   = np.load("../data/datasets/mnist_14x14_ytrn.npy")
#xtst = np.load("../data/datasets/mnist_14x14_xtst.npy")/256.0
#yt   = np.load("../data/datasets/mnist_14x14_ytst.npy")

nc = yn.max() + 1
ytrn = np.zeros((len(yn),nc), dtype="uint8")
for i in range(nc):
    j = np.where(yn == i)[0]
    ytrn[j,i] = 1

ytst = np.zeros((len(yt),nc), dtype="uint8")
for i in range(nc):
    j = np.where(yt == i)[0]
    ytst[j,i] = 1

# define and test the model
model = train(xtrn, ytrn, nodes)
prob = predict(xtst,model)
cm,acc = confusion(prob,ytst)

print(cm)
print()
print("accuracy = %0.6f" % acc)
print()

