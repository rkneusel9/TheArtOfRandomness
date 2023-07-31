#
#  file:  store.py
#
#  RTK, 03-Jun-2022
#  Last update:  03-Jun-2022
#
################################################################

import pickle
import sys
import os
import numpy as np
import time
import matplotlib.pylab as plt

sys.path.append("../include/")

from PSO import *
from RO import *
from GWO import *
from Jaya import *
from GA import *
from DE import *

from Bounds import *
from RandomInitializer import *
from LinearInertia import *
from RE import *


################################################################
#  Select
#
def Select(fi, rng):
    """Select a product according to these frequencies"""

    t = rng.random()
    c = 0.0
    for i in range(len(fi)):
        c += fi[i]
        if (c >= t):
            return i


################################################################
#  Shopper
#
class Shopper:
    """Represent a single shopper"""

    def __init__(self, fi, pv, rng):
        """Constructor"""
        self.item_values = pv

        #  The shopper wants to buy this product
        self.target = Select(fi,rng)

        #  Now select others the shopper will buy
        #  if encountered
        self.impulse = np.argsort(rng.random(len(fi)))[:3]
        while (self.target in self.impulse):
            self.impulse = np.argsort(rng.random(len(fi)))[:3]

    def GoShopping(self, products):
        """Go shopping and return the amount spent"""
        
        spent = 0.0
        for p in products:
            if (p == self.target):
                spent += self.item_values[p]
                break
            if (p in self.impulse):
                spent += self.item_values[p]
        return spent


################################################################
#  Objective
#
class Objective:
    """Simulate a day's worth of customers"""

    def __init__(self, nshoppers, pci, pv, rng):
        """Constructor"""

        self.nshoppers = nshoppers
        self.fcount = 0
        
        self.shoppers = []
        for i in range(nshoppers):
            shopper = Shopper(pci, pv, rng)
            self.shoppers.append(shopper)

    def Evaluate(self, p):
        """Evaluate an arrangement of products"""

        self.fcount += 1
        order = np.argsort(p)
        revenue = 0.0
        for i in range(self.nshoppers):
            revenue += self.shoppers[i].GoShopping(order)
        return -revenue


#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("store <nshoppers> <npart> <niter> <alg> <kind>")
    print()
    print("  <nshoppers> - number of shoppers per day")
    print("  <npart>     - number of particles")
    print("  <niter>     - number of iterations")
    print("  <alg>       - PSO,RO,GWO,JAYA,GA,DE,BARE")
    print("  <kind>      - randomness source")
    print()
    exit(0)

products = pickle.load(open("products.pkl","rb"))
nshoppers = int(sys.argv[1])
npart = int(sys.argv[2])
niter = int(sys.argv[3])
alg = sys.argv[4].upper()
kind = sys.argv[5]

#  Item names and frequencies
ci = products[0]  # product counts
ni = products[1]  # product names
pv = products[2]  # product values
pci = ci / ci.sum()  # probability of being purchased
N = len(ci)          # number of products

ndim = len(ci)
rng = RE(kind=kind)
b = Bounds([0]*ndim, [1]*ndim, enforce="resample", rng=rng)
i = RandomInitializer(npart, ndim, bounds=b, rng=rng)
obj = Objective(nshoppers, pci, pv, rng)

if (alg == "PSO"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, max_iter=niter, bounds=b, rng=rng, inertia=LinearInertia())
elif (alg == "BARE"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, max_iter=niter, bounds=b, bare=True, rng=rng)
elif (alg == "RO"):
    swarm = RO(obj=obj, npart=npart, ndim=ndim, init=i, max_iter=niter, bounds=b, rng=rng)
elif (alg == "GWO"):
    swarm = GWO(obj=obj, npart=npart, ndim=ndim, init=i, max_iter=niter, bounds=b, rng=rng)
elif (alg == "JAYA"):
    swarm = Jaya(obj=obj, npart=npart, ndim=ndim, init=i, max_iter=niter, bounds=b, rng=rng)
elif (alg == "GA"):
    swarm = GA(obj=obj, npart=npart, ndim=ndim, init=i, max_iter=niter, bounds=b, rng=rng)
elif (alg == "DE"):
    swarm = DE(obj=obj, npart=npart, ndim=ndim, init=i, max_iter=niter, bounds=b, rng=rng)
else:
    raise ValueError("Unknown algorithm: %s" % alg)

st = time.time()
swarm.Optimize()
en = time.time()

res = swarm.Results()

print()
print("Maximum daily revenue $%0.2f (time %0.3f seconds)" % (-res["gbest"][-1], en-st))
print("(%d best updates, %d function evaluations)" % (len(res["gbest"]), obj.fcount))
print()
print("Product order:")
order = np.argsort(res["gpos"][-1])
ni = ni[order]
pci= pci[order]
pv = pv[order]
for p in range(len(pv)):
    print("%25s  (%4.1f%%) ($%0.2f)" % (ni[p], 100.0*pci[p], pv[p]))
    if (ni[p] == "whole milk"):
        milk_rank = p
    if (ni[p] == "candy"):
        candy_rank = p
print()
print("milk rank = %d" % milk_rank)
print("candy rank = %d" % candy_rank)
print()
print("Upper half median probability of being selected = %4.1f" % (100.0*np.median(pci[:N//2]),))
print("                           median product value = %4.2f" % (np.median(pv[:N//2]),))
print("Lower half median probability of being selected = %4.1f" % (100.0*np.median(pci[N//2:]),))
print("                           median product value = %4.2f" % (np.median(pv[N//2:]),))
print()


