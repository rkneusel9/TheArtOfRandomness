#
#  file:  gp.py
#
#  Genetic programming to fit code to data.
#
#  RTK, 16-May-2022
#  Last update:  19-May-2022
#
################################################################

import os
import time
import numpy as np
import matplotlib.pylab as plt
from GA import *
from RO import *
from MiCRO import *
from DE import *
from PSO import *
from Jaya import *
from LinearInertia import *
from Bounds import *
from RandomInitializer import *
from RE import *

################################################################
#  GetData
#
def GetData(fname):
    """Read a data file"""
    d = np.loadtxt(fname)
    return d[:,0], d[:,1]


################################################################
#  Number
#
def Number(f, gmin=-20.0, gmax=20.0):
    """Convert fraction to number"""
    return gmin + f*(gmax-gmin)
    

################################################################
#  StrExpression
#
def StrExpression(expr, gmin=-20.0, gmax=20.0):
    """Return a string version of the expression"""

    operations = [
        "add","sub","mul","div","mod","pow","neg","push(x)","halt"]

    s = ["push(x)"]
    for e in expr:
        if (e < 1.0):
            s.append("push(%0.5f)" % Number(e, gmin=gmin, gmax=gmax))
        else:
            #  an operation
            op = int(np.floor(e))
            name = operations[op-1] 
            s.append(operations[op-1])
    return s


################################################################
#  Expression
#
def Expression(x, expr, gmin=-20.0, gmax=20.0):
    """Evaluate the expression in expr"""

    def BinaryOp(s,op):
        """Apply a binary operation"""
        b = s.pop()
        a = s.pop()
        if (op == 0):
            c = a + b
        elif (op == 1):
            c = a - b
        elif (op == 2):
            c = a * b
        elif (op == 3):
            c = a / b
        elif (op == 4):
            c = a % b
        elif (op == 5):
            c = a**b
        s.append(c)

    bad = 1e9   # return on bad expressions
    s = [x]     # initial stack

    try:
        for e in expr:
            if (e < 1.0):
                s.append(Number(e, gmin=gmin, gmax=gmax))
            else:
                #  an operation
                op = int(np.floor(e))
                if (op < 7):
                    BinaryOp(s, op-1)
                elif (op == 7):
                    s.append(-s.pop())
                elif (op == 8):
                    s.append(x)
                elif (op == 9):
                    break  # halt
    except:
        return bad  # bad expression
    
    try:
        return s.pop()
    except:
        return bad  # empty stack, also bad


################################################################
#  Objective function
#
class Objective:
    """Objective function"""

    def __init__(self, x,y, gmin=-20.0, gmax=20.0):
        """Constructor"""

        self.fcount = 0
        self.x = x.copy()
        self.y = y.copy()
        self.gmin = gmin
        self.gmax = gmax

    def Evaluate(self, p):
        """Calculate the MSE"""

        self.fcount += 1
        y = np.zeros(len(self.x))
        for i in range(len(self.x)):
            y[i] = Expression(self.x[i],p, self.gmin, self.gmax)
            if (np.isnan(y[i])):
                y[i] = 1e9
        return ((y - self.y)**2).mean()


#
#  main:
#
if (len(sys.argv) == 1):
    print()
    print("gp <data> <lo> <hi> <length> <npart> <niter> <alg> <source> [<plot>]")
    print()
    print("  <data>   -  dataset, y then x")
    print("  <lo>,<hi>-  min/max range for constants")
    print("  <length> -  program length (ndim)")
    print("  <npart>  -  number of swarm particles")
    print("  <niter>  -  number of iterations")
    print("  <alg>    -  algorithm")
    print("  <source> -  randomness source")
    print("  <plot>   -  output fit plot name")
    print()
    exit(0)

#  Parse the command line
Y,X = GetData(sys.argv[1])
lo = float(sys.argv[2])
hi = float(sys.argv[3])
ndim = int(sys.argv[4])
npart= int(sys.argv[5])
niter= int(sys.argv[6])
alg = sys.argv[7].upper()
kind = sys.argv[8]

#  Set up the search
rng = RE(kind=kind)
b = Bounds([0]*ndim,[10]*ndim, enforce="resample", rng=rng)
i = RandomInitializer(npart, ndim, bounds=b, rng=rng)
obj = Objective(X,Y,lo,hi)

if (alg == "GA"):
    swarm = GA(obj=obj, npart=npart, ndim=ndim, init=i, tol=0, max_iter=niter, bounds=b, rng=rng, top=0.2)
elif (alg == "DE"):
    swarm = DE(obj=obj, npart=npart, ndim=ndim, init=i, tol=0, max_iter=niter, bounds=b, rng=rng)
elif (alg == "JAYA"):
    swarm = Jaya(obj=obj, npart=npart, ndim=ndim, init=i, tol=0, max_iter=niter, bounds=b, rng=rng)
elif (alg == "PSO"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=0, max_iter=niter, bounds=b, rng=rng,
                vbounds=Bounds([-10]*ndim, [10]*ndim, enforce="clip", rng=rng), 
                inertia=LinearInertia(), ring=True, neighbors=6)
elif (alg == "BARE"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=0, max_iter=niter, bounds=b, rng=rng, bare=True)
elif (alg == "RO"):
    swarm = RO(obj=obj, npart=npart, ndim=ndim, init=i, tol=0, max_iter=niter, bounds=b, rng=rng)
elif (alg == "MICRO"):
    swarm = MiCRO(obj=obj, npart=npart, ndim=ndim, init=i, tol=0, max_iter=niter, bounds=b, rng=rng)
else:
    raise ValueError("Unknown algorithm: %s" % alg)

#  Do the search
st = time.time()
swarm.Optimize()
en = time.time()

#  Get the results
res = swarm.Results()

print()
print("Minimum mean total squared error: %0.9f  (%s)" % (res["gbest"][-1],os.path.basename(sys.argv[1])))
expr = res["gpos"][-1]
for s in StrExpression(expr,lo,hi):
    print("    %s" % s)
print("(%d best updates, %d function calls, time: %0.3f seconds)" % (len(res["gbest"]), swarm.obj.fcount, en-st))
print()

if (len(sys.argv) > 9):
    x = np.linspace(X.min(),X.max(),200)
    y = []
    for i in range(len(x)):
        y.append(Expression(x[i],expr,lo,hi))
    y = np.array(y)
    plt.plot(x,y,color='b')
    plt.plot(X,Y,color='r',marker='.',linestyle='none')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(sys.argv[9], dpi=300)

