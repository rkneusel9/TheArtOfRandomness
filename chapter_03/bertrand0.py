#
#  file:  bertrand0.py
#
#  Random points on the circumference approach
#
#  RTK, 23-Apr-2022
#  Last update:  23-Apr-2023
#
################################################################

from RE import *
import numpy as np
import matplotlib.pylab as plt
import sys

if (len(sys.argv) == 1):
    print()
    print("bertrand0 <npts> <output> pcg64|mt19937|minstd|urandom|rdrand|<file> [<seed>]")
    print()
    print("  <npts>   - number of points to simulate")
    print("  <output> - output image name")
    print("  <seed>   - PRNG seed (optional)")
    print()
    exit(0)

N = int(sys.argv[1])
oname = sys.argv[2]
kind = sys.argv[3]
if (len(sys.argv) == 5):
    seed = int(sys.argv[4])
    rng = RE(kind=kind, seed=seed)
else:
    rng = RE(kind=kind)

r = 1
l = 2*r*np.cos(np.pi/6)
cs = np.cos(np.pi/2)
sn = np.sin(np.pi/2)

X = []
Y = []
for i in range(2*N):
    x = -r + 2*r*rng.random()
    s = (-1)**np.floor(2*rng.random())
    y = s*np.sqrt(r**2 - x**2)
    if (rng.random() < 0.5):
        a = x*cs - y*sn
        b = x*sn + y*cs
        x = a; y = b
    X.append(x)
    Y.append(y)

#  Calculate the probability that the chord is longer
#  than the side of the equilateral triangle
M = k = 0
while (k < len(X)):
    d = np.sqrt((X[k]-X[k+1])**2 + (Y[k]-Y[k+1])**2)
    M += 1 if (d > l) else 0
    k += 2
print("Probability is approximately %d/%d = %0.7f" % (M,N, M/N))

#  Plot no more than 600 chords
n = 0
k = 0
while (k < len(X)):
    if (n < 600):
        plt.plot([X[k],X[k+1]],[Y[k],Y[k+1]],linewidth=0.5,color='k')
        n += 1
    k += 2
plt.axis('equal')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname, dpi=300)
plt.show()



