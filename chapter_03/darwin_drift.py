#
#  file: darwin_drift.py
#
#  Simulate evolution of simple organisms that need to adapt
#  to their static environment and genetic drift
#
#  RTK, 16-Apr-2022
#  Last update:  20-Apr-2022
#
################################################################

import sys
import numpy as np
import matplotlib.pylab as plt
from RE import *
from PIL import Image


################################################################
#  MakeRGB
#
def MakeRGB(pop):
    """Convert a population to an image row"""
    
    r = []; g = []; b = []
    for i in range(len(pop)):
        r.append((int(16*pop[i,0]) << 4) + int(16*pop[i,1]))
        g.append((int(16*pop[i,2]) << 4) + int(16*pop[i,3]))
        b.append((int(16*pop[i,4]) << 4) + int(16*pop[i,5]))
    row = np.zeros((len(pop),3), dtype="uint8")
    row[:,0] = np.array(r)
    row[:,1] = np.array(g)
    row[:,2] = np.array(b)
    return row


################################################################
#  MakeEnvRGB
#
def MakeEnvRGB(env, cols):
    """Convert an environment to an image row"""
    
    r = []; g =[]; b = []
    for i in range(cols):
        r.append((int(16*env[0]) << 4) + int(16*env[1]))
        g.append((int(16*env[2]) << 4) + int(16*env[3]))
        b.append((int(16*env[4]) << 4) + int(16*env[5]))
    row = np.zeros((cols,3), dtype="uint8")
    row[:,0] = np.array(r)
    row[:,1] = np.array(g)
    row[:,2] = np.array(b)
    return row


################################################################
#  Mate
#
def Mate(pop, fitness, advantage):
    a = advantage / 1000
    i = int(len(pop)*np.random.beta(1,1+a))
    j = i
    while (j == i):
        j = int(len(pop)*np.random.beta(1,1+a))
    c = int(6*rng.random())
    org = np.hstack((pop[i][:c], pop[j][c:]))
    
    #  Small chance of mutation
    if (rng.random() < mutation):
        c = int(6*rng.random())
        org[c] = int(16*rng.random())
    return org


################################################################
#  Main
#
if (len(sys.argv) == 1):
    print()
    print("darwin_drift <generations> <advantage> <mutation> <good> <fraction> pcg64|mt19937|minstd|urandom|rdrand|<file> <output> [<seed>]")
    print()
    print("  <generations>  -  number of generations to simulate")
    print("  <advantage>    -  advantage to fitter organisms (int, [0,1000])")
    print("  <mutation>     -  probability of mutation (float, [0,1])")
    print("  <good>         -  'good enough' minimum score (e.g. 4)")
    print("  <fraction>     -  subgroup size, fraction [0,1]")
    print("  <output>       -  output base name (no extension)")
    print("  <seed>         -  PRNG seed (optional)")
    print()
    exit(0)

ngen = int(sys.argv[1])
advantage = int(sys.argv[2])
mutation = float(sys.argv[3])
good = float(sys.argv[4])
frac = float(sys.argv[5])
kind = sys.argv[6]
oname = sys.argv[7]

if (len(sys.argv) == 9):
    seed = int(sys.argv[8])
    rng = RE(kind=kind, seed=seed)
else:
    rng = RE(kind=kind)

#  Initial population
npop = 384
pop = np.zeros((npop, 6))
for i in range(npop):
    pop[i,:] = (16*rng.random(6)).astype("uint8")

#  Environment -- changes once
environment = (16*rng.random(6)).astype("uint8")
gen0 = int(0.5*ngen*rng.random())
gen1 = int(0.4*ngen*rng.random()) + gen0

#  Generations
ndrift = int(frac*npop)
hpop = np.zeros((ngen,npop,6))
henv = np.zeros((ngen,6))

#  Track population fitness to plot at end of the run
pfitg = []
pfit0 = []
pfit1 = []

for g in range(ngen):
    if (g < gen0):
        #  Fitness
        fitness = np.zeros(npop)
        for i in range(npop):
            d = np.sqrt(((pop[i]-environment)**2).sum())
            if (d < good):
                d = good
            fitness[i] = d
        
        idx = np.argsort(fitness)
        pop = pop[idx]
        fitness = fitness[idx]

        #  History
        hpop[g,:,:] = pop
        henv[g,:] = environment

        #  Average fitness of the population
        pfitg.append(fitness.mean())
        print("%6d: fitness = %0.8f" % (g, pfitg[-1]))

        #  Next generation
        nxt = []
        for i in range(npop):
            nxt.append(Mate(pop,fitness,advantage))
        pop = np.array(nxt)
    else:
        #  Split the new generation.  Note, it isn't sorted
        #  by fitness yet
        pop0 = pop[:ndrift]
        pop1 = pop[ndrift:]

        fit0 = np.zeros(len(pop0))
        for i in range(len(pop0)):
            d = np.sqrt(((pop0[i]-environment)**2).sum())
            if (d < good):
                d = good
            fit0[i] = d
        idx = np.argsort(fit0)
        pop0 = pop0[idx]
        fit0 = fit0[idx]

        fit1 = np.zeros(len(pop1))
        for i in range(len(pop1)):
            d = np.sqrt(((pop1[i]-environment)**2).sum())
            if (d < good):
                d = good
            fit1[i] = d
        idx = np.argsort(fit1)
        pop1 = pop1[idx]
        fit1 = fit1[idx]

        #  History
        hpop[g,:ndrift,:] = pop0
        hpop[g,ndrift:,:] = pop1
        henv[g,:] = environment

        #  Average fitness of the populations
        pfit0.append(fit0.mean())
        pfit1.append(fit1.mean())
        print("%6d: fit0 = %0.8f, fit1 = %0.8f" % (g, pfit0[-1], pfit1[-1]))

        #  Next generation
        nxt = []
        for i in range(len(pop0)):
            nxt.append(Mate(pop0,fit0,advantage))
        pop0 = np.array(nxt)
        nxt = []
        for i in range(len(pop1)):
            nxt.append(Mate(pop1,fit1,advantage))
        pop1 = np.array(nxt)
        pop = np.vstack((pop0,pop1))

    #  Time to change environment?
    if (g == gen1):
        environment = (16*rng.random(6)).astype("uint8")

#  Create the output image
imgp = np.zeros((ngen, npop, 3), dtype="uint8")
for i in range(ngen):
    imgp[i,:,:] = MakeRGB(hpop[i,:,:])

ecol = 8
imge = np.zeros((ngen, ecol, 3), dtype="uint8")
for i in range(ngen):
    imge[i,:,:] = MakeEnvRGB(henv[i,:], ecol)

blank = np.zeros((ngen, 2, 3), dtype="uint8")
img = np.hstack((imge,blank,imgp))

image = Image.fromarray(img)
rows, cols = image.size
image = image.resize((3*rows, 3*cols), Image.NEAREST)
image.save(oname+".png")

#  And the population fitness plot
xg = np.arange(gen0)
plt.plot(xg,pfitg,color='k')
xf = np.arange(ngen-gen0) + gen0
plt.plot(xf,pfit0,color='k', linewidth=2)
plt.plot(xf,pfit1,color='k', linewidth=1)
plt.xlabel("Generation")
plt.ylabel("Mean Population Fitness")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+"_plot.png", dpi=300)
plt.savefig(oname+"_plot.eps", dpi=300)

