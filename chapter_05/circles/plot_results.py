#
#  file: plot_results.py
#
#  Plot each algorithm's results for the given
#  number of circles.
#
#  RTK, 25-May-2022
#  Last updated:  25-May-2022
#
################################################################

import numpy as np
import matplotlib.pylab as plt
import sys
import os

def PlotCircle(a,b,r):
    """Plot a circle of radius r centered at a,b"""

    x = np.linspace(-r,r,300)
    y0 = np.sqrt(r**2-x**2)
    y1 = -np.sqrt(r**2-x**2)
    x = np.hstack((x,np.linspace(-r,r,300)))
    y = np.hstack((y0,y1))
    n = np.hstack((x,y))
    m = np.hstack((y,x))
    x = n + a
    y = m + b
    plt.plot([a,a],[b,b], marker='+', color='k', linestyle='none')
    plt.plot(x,y, marker='.', markersize=0.5, color='k', linestyle='none')


def PlotCircles(X,Y,n,alg):
    """Plot circles at the given centers using the known best radius"""
    
    #  Table D1, Croft 1991
    d = [np.sqrt(2), np.sqrt(6)-np.sqrt(2), 1, np.sqrt(2)/2,
         np.sqrt(13)/6, 2*(2-np.sqrt(3)), (np.sqrt(6)-np.sqrt(2))/2, 0.5,
         0.42127954,0.398,np.sqrt(34)/15,(np.sqrt(3)-1)/2,(np.sqrt(6)-np.sqrt(2))/3,
         4/(8+np.sqrt(2)+np.sqrt(6)),1/3,0.306,np.sqrt(13)/12,.290,(6-np.sqrt(2))/16]
    r = d[n-2]/2

    for i in range(len(X)):
        PlotCircle(X[i],Y[i],r)

    plt.axis('equal')
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig("plots/circles_%s_%d.png" % (alg,n), dpi=300)
    #plt.savefig("plots/circles_%s_%d.eps" % (alg,n), dpi=300)
    plt.close()


if (len(sys.argv) == 1):
    print()
    print("plot_results <n> <base_dir>")
    print()
    print("<n>        - 2..9, number of circles")
    print("<base_dir> - base directory name")
    print()
    exit(0)

n = int(sys.argv[1])
base = sys.argv[2]
algs = ["ro","ga","bare","pso","de","gwo","jaya"]

for m,alg in enumerate(algs):
    X = []
    Y = []
    lines = [i[:-1] for i in open(base+("/%s%d/README.txt" % (alg,n))) if i.find("x,y") != -1]

    for line in lines:
        x,y = [k.strip() for k in line.split("=")[1].split(",")]
        x = float(x[1:])
        y = float(y[:-1])
        X.append(x)
        Y.append(y)
    PlotCircles(X,Y,n,alg)

