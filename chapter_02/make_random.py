#
#  file:  make_random.py
#
#  Generate files of random bytes
#
#  RTK, 10-Apr-2022
#  Last update:  10-Apr-2022
#
################################################################

import numpy as np
from RE import *
import sys

if (len(sys.argv) == 1):
    print()
    print("make_random <n> <seed> <output> [pcg64|mt19937|minstd|rdrand|urandom|quasi<b>]")
    print()
    print("  <n>      - number of bytes")
    print("  <seed>   - integer or 'none'")
    print("  <output> - output filename")
    print("  optional: generator source, <b> is base")
    print()
    exit(0)

N = int(sys.argv[1])
seed = sys.argv[2]
ofile = sys.argv[3]
gen = "pcg64"
if (len(sys.argv) == 5):
    gen = sys.argv[4]
    if (gen[:5] == "quasi"):
        b = int(gen[5:])
        gen = "quasi"

if (seed == "none"):
    if (gen == "quasi"):
        rng = RE(kind=gen, mode="byte", base=b)
    else:
        rng = RE(kind=gen, mode="byte")
else:
    if (gen == "quasi"):
        rng = RE(kind=gen, mode="byte", base=b, seed=int(seed))
    else:
        rng = RE(kind=gen, mode="byte", seed=int(seed))

rng.random(N).tofile(ofile)

