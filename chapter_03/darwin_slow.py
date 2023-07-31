#
#  file: darwin_slow.py
#
#  Simulate evolution of simple organisms that need to adapt
#  to slow environmental changes
#
#  RTK, 16-Apr-2022
#  Last update:  19-Apr-2022
#
################################################################

import sys
import numpy as np
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
    print("darwin_slow <generations> <advantage> <mutation> <good> <eprob> pcg64|mt19937|minstd|urandom|rdrand|<file> <output> [<seed>]")
    print()
    print("  <generations>  -  number of generations to simulate")
    print("  <advantage>    -  advantage to fitter organisms (int, [0,1000])")
    print("  <mutation>     -  probability of mutation (float, [0,1])")
    print("  <good>         -  'good enough' minimum score (e.g. 4)")
    print("  <eprob>        -  probability of environmental change (e.g. 0.01)")
    print("  <output>       -  output image name (.png extension)")
    print("  <seed>         -  PRNG seed (optional)")
    print()
    exit(0)

ngen = int(sys.argv[1])
advantage = int(sys.argv[2])
mutation = float(sys.argv[3])
good = float(sys.argv[4])
eprob = float(sys.argv[5])
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

#  Environment -- dynamic
environment = (16*rng.random(6)).astype("uint8")

#  Generations
hpop = np.zeros((ngen,npop,6))
henv = np.zeros((ngen,6))

for g in range(ngen):
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
    print("%6d: fitness = %0.8f" % (g, fitness.mean()))

    #  Next generation
    nxt = []
    for i in range(npop):
        nxt.append(Mate(pop,fitness,advantage))
    pop = np.array(nxt)
    
    #  Option to change the environment slowly
    if (rng.random() < eprob):
        offset = 2*rng.random(6)-1
        environment = environment + offset
        environment = np.maximum(0,np.minimum(15,environment))
        environment = (environment + 0.5).astype("uint8")

#  Create the output image
imgp = np.zeros((ngen, npop, 3), dtype="uint8")
for i in range(ngen):
    imgp[i,:,:] = MakeRGB(hpop[i,:,:])

ecol = 8
imge = np.zeros((ngen, ecol, 3), dtype="uint8")
for i in range(ngen):
    e = MakeEnvRGB(henv[i,:], ecol)
    if (i > 0):
        old = MakeEnvRGB(henv[i-1,:], ecol)
        if (not np.array_equal(e, old)):
            continue
    imge[i,:,:] = MakeEnvRGB(henv[i,:], ecol)

blank = np.zeros((ngen, 2, 3), dtype="uint8")
img = np.hstack((imge,blank,imgp))

image = Image.fromarray(img)
rows, cols = image.size
image = image.resize((3*rows, 3*cols), Image.NEAREST)
image.save(oname)


