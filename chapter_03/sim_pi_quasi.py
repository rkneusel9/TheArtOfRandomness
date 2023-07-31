#
#  file:  sim_pi_quasi.py
#
#  "A Slice of Pi" -- simulate pi using quasirandom numbers
#
#  RTK, 15-Apr-2022
#  Last update:  15-Apr-2022
#
################################################################

import sys
from RE import *

def Simulate(N, rng0, rng1):
    """Use random numbers to estimate pi"""

    inside = 0
    for i in range(N):
        x = rng0.random()
        y = rng1.random()
        if (x*x + y*y) <= 1.0:
            inside += 1
    return 4.0*inside/N

if (len(sys.argv) == 1):
    print()
    print("sim_pi <n> <base0> <base1>")
    print()
    print("  <n>      - number of samples")
    print("  <base0>  - first base")
    print("  <base1>  - second base")
    print()
    exit(0)

N = int(sys.argv[1])
base0 = int(sys.argv[2])
base1 = int(sys.argv[3])

rng0 = RE(kind="quasi", base=base0)
rng1 = RE(kind="quasi", base=base1)
pi = Simulate(N, rng0, rng1)

print("pi = %0.8f" % pi)

