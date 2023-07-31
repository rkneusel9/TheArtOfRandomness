import os
import sys
from RE import *
from ifs import *

if (len(sys.argv) == 1):
    print()
    print("ifs_maps <n> <outdir> <seed>")
    print()
    print("  <n>      - number of random fractals")
    print("  <outdir> - output directory")
    print("  <seed>   - random number seed")
    print()
    exit(0)

N = int(sys.argv[1])
outdir = sys.argv[2]
seed = int(sys.argv[3])

rng = RE(kind="minstd", seed=seed)
os.system("mkdir %s 2>/dev/null" % outdir)

for k in range(N):
    oname = "%s/fractal_%04d.png" % (outdir, k)
    mseed = int(9999 + 9999999*rng.random())
    r = int(256*rng.random())
    g = int(256*rng.random())
    b = int(256*rng.random())
    color = "%02x%02x%02x" % (r,g,b)
    prng = RE(kind="mt19937", seed=mseed)
    app = IFS(1_000_000, "random", color, prng)
    app.GeneratePoints()
    app.StoreFractal(oname)
    print("%4d: seed %d" % (k, mseed))

