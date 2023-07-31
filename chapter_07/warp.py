#
#  file:  warp.py
#
#  Random images via warping the [-1,1] grid
#
#  RTK, 29-Jun-2022
#  Last update:  29-Jun-2022
#
################################################################

import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from RE import *

#
#  Pixel functions:
#
def f0(a,b):
    x,y = a**3 + b, b**2 + a
    c = int(255*(a*b+1)/2)
    return x,y,c

def f1(a,b):
    x,y = b**3 + a, a**2 + b
    c = int(255*(a*b+1)/2)
    return x,y,c

def f2(a,b):
    x,y = b*a**2, a*b**2
    c = int(255*(a+1)/2)
    return x,y,c

def f3(a,b):
    x,y = a*b**2, b*a**2
    c = int(255*(b+1)/2)
    return x,y,c

def f4(a,b):
    x,y = a*np.exp(b), b*np.exp(a)
    c = int(255*(a*b+1)/2)
    return x,y,c

funcs = [f0,f1,f2,f3,f4]

#
#  Main:
#
if (len(sys.argv) == 1):
    print()
    print("warp <points> <cycles> <output> [<kind> | <kind> <seed>]")
    print()
    print("  <points> - points along each grid axis (e.g. 200)")
    print("  <cycles> - number of iterations (e.g. 1)")
    print("  <output> - output filename")
    print("  <kind>   - randomness source")
    print("  <seed>   - seed value")
    print()
    exit(0)

npoints = int(sys.argv[1])
cycles = int(sys.argv[2])
oname = sys.argv[3]

showPlot = True
if (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[4])
elif (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[4], seed=int(sys.argv[5]))
elif (len(sys.argv) == 7):
    rng = RE(kind=sys.argv[4], seed=int(sys.argv[5]))
    showPlot = False
else:
    rng = RE()

cnames = [i[:-1].strip() for i in open("color_map_names.txt")]
X = []; Y = []; C = []
v = -1 + 2*np.arange(npoints)/npoints


for k in range(cycles):
    n = int(len(cnames)*rng.random())
    cmap = cm.get_cmap(cnames[n])
    n = int(len(funcs)*rng.random())
    fn = funcs[n]
    xoff,yoff = rng.random(2)-0.5
    theta = np.pi*rng.random()
    for i in range(len(v)):
        for j in range(len(v)):
            n,m,c = fn(v[i],v[j])
            x = n*np.cos(theta) - m*np.sin(theta)
            y = n*np.sin(theta) + m*np.cos(theta)
            X.append(x+xoff)
            Y.append(y+yoff)
            C.append(cmap(c))

plt.scatter(X,Y, marker=',', s=0.6, c=C)
plt.axis('off')
plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.savefig(oname, dpi=300)
if showPlot:
    plt.show()

