#
#  file:  spheres.py
#
#  Optimize placing points in the unit cube, which is
#  equivalent to packing spheres of the same radius.
#
#  RTK, 25-May-2022
#  Last update:  29-May-2022
#
################################################################

import pickle
import time
import os
import sys
import numpy as np
import matplotlib.pylab as plt

sys.path.append("../include/")

from DE import *
from RO import *
from PSO import *
from GWO import *
from Jaya import *
from GA import *
from RE import *

from Bounds import *
from RandomInitializer import *
from LinearInertia import *


################################################################
#  Objective
#
class Objective:
    """Determine coverage area"""

    def __init__(self):
        """Constructor"""

        self.fcount = 0

    def Evaluate(self, p):
        """Evaluate a set of positions"""

        self.fcount += 1
        n = p.shape[0]//3
        xy = p.reshape((n,3))
        
        # Find the minimal point separation
        dmin = 10.0
        for i in range(n):
            for j in range(i,n):
                if (i==j):
                    continue
                d = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2 + (xy[i,2]-xy[j,2])**2)
                if (d < dmin):
                    dmin = d
        
        return -dmin


################################################################
#  main
#
def main():
    """Place points in the unit square"""

    if (len(sys.argv) == 1):
        print()
        print("points <n> <npart> <niter> <alg> <kind> <outdir>")
        print()
        print("  <n>       -  number of points to place")
        print("  <npart>   -  number of swarm particles")
        print("  <niter>   -  number of swarm iterations")
        print("  <alg>     -  DE|RO|PSO|GWO|JAYA|GA|BARE")
        print("  <kind>    -  randomness source")
        print("  <outdir>  -  output directory (overwritten)")
        print()
        return

    ndim = 3*int(sys.argv[1])
    npart = int(sys.argv[2])
    niter = int(sys.argv[3])
    alg = sys.argv[4].upper()
    kind = sys.argv[5]
    outdir = sys.argv[6]

    os.system("rm -rf %s; mkdir %s" % (outdir,outdir))

    rng = RE(kind=kind)
    b = Bounds([0]*ndim, [1]*ndim, enforce="clip", rng=rng)
    i = RandomInitializer(npart, ndim, bounds=b, rng=rng)

    obj = Objective()

    if (alg == "PSO"):
        swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, rng=rng, inertia=LinearInertia())
    elif (alg == "BARE"):
        swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, bare=True, rng=rng)
    elif (alg == "DE"):
        swarm = DE(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, rng=rng)
    elif (alg == "RO"):
        swarm = RO(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, rng=rng)
    elif (alg == "GWO"):
        swarm = GWO(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, rng=rng)
    elif (alg == "JAYA"):
        swarm = Jaya(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, rng=rng)
    elif (alg == "GA"):
        swarm = GA(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, rng=rng)

    st = time.time()
    swarm.Optimize()
    en = time.time()

    res = swarm.Results()
    pickle.dump(res, open(outdir+"/results.pkl","wb"))

    s  = "\nSearch results: %s, %d particles, %d iterations\n\n" % (alg, npart, niter)
    s += "Optimization minimum %0.8f (time = %0.3f)\n" % (res["gbest"][-1], en-st)
    s += "(%d best updates, %d function evaluations)\n\n" % (len(res["gbest"]), obj.fcount)

    #  Generate the output plot
    p = res["gpos"][-1]
    n = p.shape[0]//3
    xy = p.reshape((n,3))

    for i in range(n):
        s += "    (x,y,z) = (%0.8f, %0.8f, %0.8f)\n" % (xy[i,0], xy[i,1], xy[i,2])
    s += "\n"

    print(s)
    with open(outdir+"/README.txt","w") as f:
        f.write(s)


if (__name__ == "__main__"):
    main()

