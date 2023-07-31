#
#  file:  discrete_test.py
#
#  Test various approaches to sampling from 1D discrete
#  distributions.
#
#  RTK, 12-Oct-2022
#  Last update:  13-Oct-2022
#
################################################################

from RE import *
from fldr import fldr_preprocess_int, fldr_sample
import numpy as np
import time
import sys
import matplotlib.pylab as plt

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
    print("discrete_test <N> [<kind> | <kind> <seed>]")
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

# A probability distribution
probabilities = [1,1,3,4,5,1,7,4,3]
prob = np.array(probabilities)
prob = prob / prob.sum()
M = len(prob)

#  Sequential
s = time.time()
z = np.array([Sequential(prob,rng) for i in range(N)])
e = time.time() - s
h = np.bincount(z, minlength=M)
print(h, ("(%0.6f s, sequential)" % e))

#  Reorder first
idx = np.argsort(prob)[::-1]
p = prob[idx]
s = time.time()
z = np.array([Sequential(p,rng) for i in range(N)])
e = time.time() - s
h = np.bincount(idx[z], minlength=M)
print(h, ("(%0.6f s, reordered)" % e))

#  Fast loaded dice roller (pip3 install fldr)
s = time.time()
x = fldr_preprocess_int(probabilities)
z = np.array([fldr_sample(x) for i in range(N)])
e = time.time() - s
h = np.bincount(z, minlength=M)
print(h, ("(%0.6f s, FLDR)" % e))

#  Expected output
print(np.round(prob*N).astype("uint32"), "expected")

#  Plot FLDR (others similar)
x = np.arange(len(prob))
plt.bar(x, prob, width=0.8, color='k', fill=False)
h = np.bincount(z, minlength=M)
h = h / h.sum()
plt.plot(x,h, color='k', marker='o')
plt.xlabel("Value")
plt.ylabel("Probability (%d samples)" % N)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("discrete_test.png", dpi=300)
plt.savefig("discrete_test.eps", dpi=300)
plt.close()


