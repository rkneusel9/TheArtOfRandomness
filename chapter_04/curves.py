#
#  file:  curves.py
#
#  Curve fitting with EA/SI.
#
#  RTK, 30-Dec-2019
#  Last update:  15-May-2022
#
################################################################

import sys
import os
import numpy as np
import time
import matplotlib.pylab as plt

from MiCRO import *
from PSO import *
from Jaya import *
from GA import *
from DE import *
from RO import *
from MiCRO import *
from Bounds import *
from RandomInitializer import *

from RE import *


################################################################
#  Objective
#
class Objective:
    """Generic objective function"""

    def __init__(self, x, y, func):
        """Constructor"""

        self.x = x
        self.y = y
        self.func = func
        self.fcount = 0

    def Evaluate(self, p):
        """Evaluate a position"""

        self.fcount += 1
        x = self.x
        y = eval(self.func)
        return ((y - self.y)**2).mean()


################################################################
#  GetBounds
#
def GetBounds(s,ndim):
    """Parse a bounds string"""

    if (s.find("x") == -1):
        try:
            n = np.ones(ndim)*float(s)
        except:
            n = None
    else:
        try:
            n = np.array([float(i) for i in s.split("x")])
        except:
            n = None
    return n


################################################################
#  GetData
#
def GetData(s):
    """Parse a data file"""

    lines = [i[:-1] for i in open(s)]
    ndim = int(lines[0])
    func = lines[1]
    d = np.zeros((len(lines[2:]),2))
    for i in range(2,len(lines)):
        d[i-2,:] = [float(k) for k in lines[i].split()]
    return d[:,1], d[:,0], func, ndim


################################################################
#  main
#
def main():
    """Fit functions to datasets"""

    if (len(sys.argv) == 1):
        print()
        print("curves <data> <lower> <upper> <npart> <niter> <tol> <alg> <source> [<plot>]")
        print()
        print("  <data>   - Text file w/coordinates (y x), function (first line) (.txt)")
        print("  <lower>  - lower parameter bounds")
        print("  <upper>  - upper parameter bounds")
        print("  <npart>  - number of particles")
        print("  <niter>  - number of iterations")
        print("  <tol>    - tolerance (quit if error less than)")
        print("  <alg>    - PSO,JAYA,GA,DE")
        print("  <source> - randomness source")
        print("  <plot>   - store fit plot")
        print()
        return

    #  Get the parameters
    X,Y,func,ndim = GetData(sys.argv[1])
    lower = GetBounds(sys.argv[2], ndim)
    upper = GetBounds(sys.argv[3], ndim)
    npart = int(sys.argv[4])
    niter = int(sys.argv[5])
    tol = float(sys.argv[6])
    alg = sys.argv[7].upper()
    kind = sys.argv[8]
    
    #  Randomness source
    rng = RE(kind=kind)

    #  Setup
    if (type(lower) is type(None)) and (type(upper) is type(None)):
        b = None
    else:
        b = Bounds(lower, upper, enforce="resample", rng=rng)

    i = RandomInitializer(npart, ndim, bounds=b, rng=rng)
    obj = Objective(X, Y, func)

    if (alg == "PSO"):
        swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, bare=True, rng=rng)
    elif (alg == "JAYA"):
        swarm = Jaya(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
    elif (alg == "GA"):
        swarm = GA(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
    elif (alg == "DE"):
        swarm = DE(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
    elif (alg == "RO"):
        swarm = RO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
    elif (alg == "MICRO"):
        swarm = MiCRO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, rng=rng)
    else:
        raise ValueError("Unknown algorithm: %s" % alg)

    st = time.time()
    swarm.Optimize()
    en = time.time()

    res = swarm.Results()

    print()
    print("Minimum mean total squared error: %0.9f  (%s)" % (res["gbest"][-1],os.path.basename(sys.argv[1])))
    print("Parameters:")
    for k,p in enumerate(res["gpos"][-1]):
        print("%2d: %21.16f" % (k,p))
    print("(%d best updates, %d function calls, time: %0.3f seconds)" % (len(res["gbest"]), swarm.obj.fcount, en-st))
    print()

    if (len(sys.argv) > 9):
        x = np.linspace(X.min(),X.max(),200)
        p = res["gpos"][-1]
        y = eval(func)
        plt.plot(x,y,color='b')
        plt.plot(X,Y,color='r',marker='.',linestyle='none')
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.savefig(sys.argv[9], dpi=300)


if (__name__ == "__main__"):
    main()

