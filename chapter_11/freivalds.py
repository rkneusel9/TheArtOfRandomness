#
#  file:  freivalds.py
#
#  Test Freivalds' algorithm
#
#  RTK, 30-Aug-2022
#  Last update:  31-Aug-2022
#
################################################################

import sys
import numpy as np
import time
from RE import *

################################################################
#  mmult -- naive matrix multiplication
#
def mmult(A,B):
    """Naive matrix multiplication"""
    I,K = A.shape
    J = B.shape[1]
    C = np.zeros((I,J), dtype=A.dtype)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                C[i,j] += A[i,k]*B[k,j]
    return C


################################################################
#  array_equal -- array equal test w/tolerance
#
def array_equal(a,b, eps=1e-7):
    """Test if two arrays are equal"""
    return np.abs(a-b).max() <= eps


#
#  Main
#
if (len(sys.argv) == 1):
    print()
    print("freivalds <N> <mode> <reps> [<kind> | <kind> <seed>]")
    print()
    print("  <N>     - matrix size (always square)")
    print("  <mode>  - 0=Freivalds', 1=naive")
    print("  <reps>  - reps of Freivalds' (ignored for others)")
    print("  <kind>  - randomness source")
    print("  <seed>  - seed")
    print()
    exit(0)

N = int(sys.argv[1])
mode = int(sys.argv[2])
reps = int(sys.argv[3])

if (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[4], seed=int(sys.argv[5]))
elif (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[4])
else:
    rng = RE()

k = 0       #  count number of times alg says equal (never really equal)
m = 1000    #  number of tests

s = time.time()
for i in range(m):
    #  Generate matrices that are never equal
    A = 100*rng.random(N*N).reshape((N,N))
    B = 100*rng.random(N*N).reshape((N,N))
    C = A@B + 0.1*rng.random(N*N).reshape((N,N))

    if (mode == 0):
        #  Freivalds'
        t = True
        for j in range(reps):
            r = (2*rng.random(N)).astype("uint8").reshape((N,1))
            t &= array_equal(mmult(A,mmult(B,r)), mmult(C,r))
    else:
        #  Naive
        t = array_equal(mmult(A,B), C)

    k += 1 if t else 0

print("%0.8f %0.3f" % (time.time()-s, k/m))

