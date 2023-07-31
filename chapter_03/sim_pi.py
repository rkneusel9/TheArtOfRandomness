#
#  file:  sim_pi.py
#
#  "A Slice of Pi" -- simulate pi
#
#  RTK, 15-Apr-2022
#  Last update:  15-Apr-2022
#
################################################################

import sys
from RE import *

def Simulate(N, rng):
    """Use random numbers to estimate pi"""

    v = rng.random(2*N)
    x = v[::2]
    y = v[1::2]
    d = x*x + y*y
    inside = len(np.where(d <= 1.0)[0])
    return 4.0*inside/N

if (len(sys.argv) == 1):
    print()
    print("sim_pi <n> <source>")
    print()
    print("  <n>      - number of samples")
    print("  <source> - pcg64|mt19937|minstd|urandom|rdrand|<filename>")
    print()
    exit(0)

N = int(sys.argv[1])
kind = sys.argv[2]

rng = RE(kind=kind)
pi = Simulate(N, rng)

print("pi = %0.8f" % pi)

