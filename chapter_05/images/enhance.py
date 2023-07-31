#
#  file:  enhance.py
#
#  Use local image enhancement similar to:
#       "Gray-level Image Enhancement By Particle Swarm Optimization"
#       (Gorai and Ghosh, 2009)
#
#  with a different objective function based on sharpness
#  and entropy.
#
#  RTK, 30-May-2022
#  Last update:  30-May-2022
#
################################################################

import pickle
import os
import sys
import time
import numpy as np
from PIL import Image, ImageFilter

sys.path.append("../include/")

from PSO import *
from DE import *
from RO import *
from GWO import *
from Jaya import *
from GA import *

from RandomInitializer import *
from LinearInertia import *
from Bounds import *
from RE import *


################################################################
#  ApplyEnhancement
#
def ApplyEnhancement(g, a,b,c,k):
    """Enhance an image"""

    def stats(g,i,j):
        rlo = max(i-1,0); rhi = min(i+1,g.shape[0])
        clo = max(j-1,0); chi = min(j+1,g.shape[1])
        v = g[rlo:rhi,clo:chi].ravel()
        if len(v) < 3:
            return v[0],1.0
        return v.mean(), v.std(ddof=1)

    #  enhanced image
    rows,cols = g.shape
    dst = np.zeros((rows,cols))

    #  enhance
    G = g.mean()
    for i in range(rows):
        for j in range(cols):
            m,s = stats(g,i,j)
            dst[i,j] = ((k*G)/(s+b))*(g[i,j]-c*m)+m**a

    #  return enhanced image
    dmin = dst.min()
    dmax = dst.max()
    return (255*(dst - dmin) / (dmax - dmin)).astype("uint8")


################################################################
#  Objective
#
class Objective:
    """Enhancement objective function"""

    def __init__(self, img):
        """Constructor"""

        self.img = img.copy()
        self.fcount = 0

    def F(self, dst):
        """Objective function from the paper"""
        
        r,c = dst.shape

        Is = Image.fromarray(dst).filter(ImageFilter.FIND_EDGES) 
        Is = np.array(Is)
        edgels = len(np.where(Is.ravel() > 20)[0])

        h = np.histogram(dst, bins=64)[0]
        p = h / h.sum()
        i = np.where(p != 0)[0]
        ent = -(p[i]*np.log2(p[i])).sum()
        
        F = np.log(np.log(Is.sum()))*(edgels/(r*c))*ent
        return F

    def Evaluate(self, p):
        """Apply a set of parameters"""

        self.fcount += 1
        a,b,c,k = p
        dst = ApplyEnhancement(self.img, a,b,c,k)
        return -self.F(dst)


#  main
if (len(sys.argv) == 1):
    print()
    print("enhance <src> <npart> <niter> <alg> <kind> <output>")
    print()
    print("  <src>    - source grayscale image")
    print("  <npart>  - number of particles")
    print("  <niter>  - number of iterations")
    print("  <alg>    - BARE,RO,DE,PSO,JAYA,GWO,GA")
    print("  <kind>   - randomness source")
    print("  <output> - output directory (overwritten)")
    print()
    exit(0)

src = sys.argv[1]
npart = int(sys.argv[2])
niter = int(sys.argv[3])
alg = sys.argv[4].upper()
kind = sys.argv[5]
outdir = sys.argv[6]

orig = np.array(Image.open(src).convert("L"))
img = orig / 256.0

os.system("rm -rf %s; mkdir %s" % (outdir,outdir))

ndim = 4  # a,b,c,k

rng = RE(kind=kind)
b = Bounds([0.0,1.0,0.0,0.5], [1.5,22,1.0,1.5], enforce="resample", rng=rng)
i = RandomInitializer(npart, ndim, bounds=b, rng=rng)

obj = Objective(img)

if (alg == "PSO"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, inertia=LinearInertia(), rng=rng)
elif (alg == "BARE"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, bounds=b, max_iter=niter, rng=rng, bare=True)
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

s = "\nIterations:\n\n"

st = time.time()
k = 0
swarm.Initialize()
while (not swarm.Done()):
    swarm.Step()
    res = swarm.Results()
    t = "    %5d: gbest = %0.8f" % (k,res["gbest"][-1])
    print(t, flush=True)
    s += t+"\n"
    k += 1
en = time.time()

res = swarm.Results()
pickle.dump(res, open(outdir+"/results.pkl","wb"))

s += "\nSearch results: %s, %d particles, %d iterations\n\n" % (alg, npart, niter)
s += "Optimization minimum %0.8f (time = %0.3f)\n" % (res["gbest"][-1], en-st)
s += "(%d best updates, %d function evaluations)\n\n" % (len(res["gbest"]), obj.fcount)

print(s)
with open(outdir+"/README.txt","w") as f:
    f.write(s)

#  Apply the enhancement
a,b,c,k = res["gpos"][-1]
dst = ApplyEnhancement(img, a,b,c,k)
Image.fromarray(dst).save(outdir+"/enhanced.png")
Image.fromarray(orig).save(outdir+"/original.png")

