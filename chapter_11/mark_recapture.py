#
#  file:  mark_recapture.py
#
#  Mark and recapture approach to estimating a population's size.
#
#  RTK, 01-Sep-2022
#  Last update:  01-Sep-2022
#
################################################################

import sys
import numpy as np
from RE import *

if (len(sys.argv) == 1):
    print()
    print("mark_recapture <pop> <mark> <reps> [<kind> | <kind> <seed>]")
    print()
    print("  <pop>  - population size (e.g. 1000)")
    print("  <mark> - number to mark (< pop)")
    print("  <reps> - number of repetitions ")
    print("  <kind> - randomness source")
    print("  <seed> - seed")
    print()
    exit(0)

npop = int(sys.argv[1])
nmark= int(sys.argv[2])
nreps= int(sys.argv[3])

if (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[4], mode="int", low=0, high=npop, seed=int(sys.argv[5]))
elif (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[4], mode="int", low=0, high=npop)
else:
    rng = RE(mode="int", low=0, high=npop)

lincoln = []
chapman = []
bayes = []

for j in range(nreps):
    pop = np.zeros(npop, dtype="uint8")

    #  Mark some members of the population
    idx = np.argsort(rng.random(npop))[:nmark]
    pop[idx] = 1

    #  Sample until at least three marked
    K = nmark
    while (True):
        idx = np.argsort(rng.random(npop))[:K]
        k = pop[idx].sum()
        if (k > 2):
            break
        K += 5

    #  Lincoln-Petersen estimate
    lincoln.append(nmark*K/k)

    #  Chapman estimate
    chapman.append((nmark+1)*(K+1)/(k+1) - 1)

    #  Bayesian estimate
    bayes.append((nmark-1)*(K-1)/(k-2))

lincoln = np.array(lincoln)
chapman = np.array(chapman)
bayes = np.array(bayes)

print()
if (nreps == 1):
    print("Lincoln-Petersen population estimate = %d" % lincoln[0])
    print("Chapman population estimate          = %d" % chapman[0])
    print("Bayesian population estimate         = %d" % bayes[0])
else:
    print("Lincoln-Petersen population estimate = %0.4f +/- %0.4f" % (lincoln.mean(), lincoln.std(ddof=1) / np.sqrt(nreps)))
    print("Chapman population estimate          = %0.4f +/- %0.4f" % (chapman.mean(), chapman.std(ddof=1) / np.sqrt(nreps)))
    print("Bayesian population estimate         = %0.4f +/- %0.4f" % (bayes.mean(), bayes.std(ddof=1) / np.sqrt(nreps)))
    np.save("lincoln.npy", lincoln)
    np.save("chapman.npy", chapman)
    np.save("bayes.npy", bayes)
print()

