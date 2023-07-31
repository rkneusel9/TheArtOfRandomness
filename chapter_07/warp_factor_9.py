#
#  for example: python3 warp_factor_9.py 3141592 100 warpings
#
import os
import sys
from RE import *

if (len(sys.argv) == 1):
    print()
    print("warp_factor_9 <global_seed> <n> <outdir>")
    print()
    print("  <global_seed> - the seed used to generate the sequence of seeds")
    print("  <n>           - the number of warp images to make")
    print("  <outdir>      - output directory")
    exit(0)

seed = int(sys.argv[1])
N = int(sys.argv[2])
outdir = sys.argv[3]

os.system("mkdir %s" % outdir)
rng = RE(kind="minstd", seed=seed)

for k in range(N):
    cycles = int(2 + 12*rng.random())
    mseed = int(9999 + 9999999*rng.random())
    oname = "%s/warp_%04d.png" % (outdir, k)
    cmd = "python3 warp.py 200 %d %s mt19937 %d hidden" % (cycles, oname, mseed)
    os.system(cmd)

