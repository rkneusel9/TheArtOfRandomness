#
#  file:  discrete_ravel.py
#
#  Example to sample from a small 2D distribution
#
#  RTK, 21-Oct-2022
#  Last update:  21-Oct-2022
#
################################################################

import sys
from RE import *
import numpy as np

################################################################
#  Sequential
#
def Sequential(probs, rng):
    """
    Return a random sample according to probs (assumed normalized).
    Uses inversion by sequential search (Kemp1981), see p 86 of
    Devroye, Non-Uniform Random Variate Generation.
    """
    k = 0
    u = rng.random()
    while u > 0:
        u -= probs[k]
        k += 1
    return k-1

#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("discrete_ravel <N> [<kind> | <kind> <seed>]")
    print()
    print("  <N>    - number of samples")
    print("  <kind> - randomness source")
    print("  <seed> - seed")
    print()
    exit(0)

N = int(sys.argv[1])

if (len(sys.argv) == 4):
    rng = RE(kind=sys.argv[2], seed=int(sys.argv[3]))
elif (len(sys.argv) == 3):
    rng = RE(kind=sys.argv[2])
else:
    rng = RE()

# A 2D probability distribution
prob2 = np.array([[0.1, 0.0, 0.1, 0.2], 
                  [0.0, 0.0, 0.1, 0.1], 
                  [0.2, 0.0, 0.0, 0.2]])
prob = prob2.ravel()

#  Sample and display the 1D distribution
z = np.array([Sequential(prob,rng) for i in range(N)])
h = np.bincount(z, minlength=len(prob))
h = h / h.sum()
print(h)
print()

#  Reshape back to the known distribution
print(h.reshape((3,4)))
print()

#  Display the first 8 samples
print(z[:8])
print()

#  Now as pairs
x,y = np.unravel_index(z[:8], prob2.shape)
print([i for i in zip(x,y)])

