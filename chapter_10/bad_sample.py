#
#  file:  bad_sample.py
#
#  Simulate sampling error
#
#  RTK, 02-Aug-2022
#  Last update:  02-Aug-2022
#
################################################################

import numpy as np
from scipy.stats import ttest_ind
from RE import *
import sys

if (len(sys.argv) == 1):
    print()
    print("bad_sample <pop_size> <sample_size> <trials> [<kind> | <kind> <seed>]")
    print()
    print("  <pop_size>    - population size")
    print("  <sample_size> - sample size (<< pop_size)")
    print("  <trials>      - number of samples to draw")
    print("  <kind>        - randomness source")
    print("  <seed>        - seed value")
    print()
    exit(0)

npop = int(sys.argv[1])
nsamp = int(sys.argv[2])
ntrial = int(sys.argv[3])

if (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[4], seed=int(sys.argv[5]))
elif (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[4])
else:
    rng = RE()

#  Generate the population where income, smoker, and drinking
#  are a function of age
def Population(npop):
    pop = []
    for i in range(npop):
        age = 20 + int(55*rng.random())
        income = int(age*200 + age*1000*rng.random())
        income = int(income/1000)
        smoker = 0
        if (rng.random() < (0.75 - age/100)):
            smoker = 1
        drink = 1.0 - age/100
        drink = int(14*drink*rng.random())
        pop.append([age, income, smoker, drink])
    return np.array(pop)

#  For one trial, show the results explicitly
if (ntrial == 1):
    pop = Population(npop)
    idx = np.argsort(rng.random(len(pop)))[:nsamp]
    sample = pop[idx,:]
    t,p = ttest_ind(pop[:,0],sample[:,0])
    print("age   : %2.2f  %2.2f  (t=% 0.4f, p=%0.5f)" % (pop[:,0].mean(), sample[:,0].mean(), t,p))
    t,p = ttest_ind(pop[:,1],sample[:,1])
    print("income: %2.2f  %2.2f  (t=% 0.4f, p=%0.5f)" % (pop[:,1].mean(), sample[:,1].mean(), t,p))
    t,p = ttest_ind(pop[:,2],sample[:,2])
    print("smoker:  %2.2f   %2.2f  (t=% 0.4f, p=%0.5f)" % (pop[:,2].mean(), sample[:,2].mean(), t,p))
    t,p = ttest_ind(pop[:,3],sample[:,3])
    print("drink :  %2.2f   %2.2f  (t=% 0.4f, p=%0.5f)" % (pop[:,3].mean(), sample[:,3].mean(), t,p))
    exit(0)

#  For many trials, use the mean distance between the population and the sample
dist = []
for trial in range(ntrial):
    pop = Population(npop)
    idx = np.argsort(rng.random(len(pop)))[:nsamp]
    sample = pop[idx,:]
    mp = pop.mean(axis=0)
    sp = sample.mean(axis=0)
    d = np.sqrt(((mp-sp)**2).sum())
    dist.append(d)
dist = np.array(dist)

print("%d %0.8f %0.8f" % (nsamp, dist.mean(), dist.std(ddof=1) / np.sqrt(ntrial)))

