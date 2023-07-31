#
#  file:  inverse.py
#
#  Use inverse transform sampling to extract samples from
#  continuous distributions
#
#  RTK, 17-Oct-2022
#  Last update:  17-Oct-2022
#
################################################################

import numpy as np
import os
import sys
import matplotlib.pylab as plt
from RE import *


#
#  main
#

if (len(sys.argv) == 1):
    print()
    print("inverse <N> <ifunc> <func> <outdir> [<kind> | <kind> <seed>]")
    print()
    print("  <N>      - number of samples")
    print('  <ifunc>  - sampling function (e.g. "1/u"), the inverse CDF of the desired PDF')
    print('  <func>   - the PDF to sample from (e.g. "1/x")')
    print("  <outdir> - output directory (overwritten)")
    print("  <kind>   - randomness source")
    print("  <seed>   - seed")
    print()
    exit(0)

N = int(sys.argv[1])
ifunc = sys.argv[2]
func = sys.argv[3]
oname = sys.argv[4]

if (len(sys.argv) == 7):
    rng = RE(kind=sys.argv[5], seed=int(sys.argv[6]))
elif (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[5])
else:
    rng = RE()

samples = np.zeros(N)

for i in range(N):
    u = rng.random()  # [0,1)
    samples[i] = eval(ifunc)

os.system("rm -rf %s; mkdir %s" % (oname,oname))

np.save(oname+"/samples.npy", samples)

h,v = np.histogram(samples, bins=100)
h = h / h.sum()
h = h / h.max()
v = 0.5*(v[1:] + v[:-1])
plt.bar(v,h, width=0.9*(v[1]-v[0]), linewidth=0.7, color='k', fill=False)

x = np.linspace(v.min(), v.max(), 1000)
y = eval(func)
y = y / y.max()
plt.plot(x,y, color='k')

plt.xlabel("$x$")
plt.ylabel("Fraction (n=%d)" % N)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+"/histogram.png", dpi=300)
plt.savefig(oname+"/histogram.eps", dpi=300)
plt.close()

