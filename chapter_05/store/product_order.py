#
#  file:  product_order.py
#
#  Evaluate conformity to decreasing product price
#
#  RTK, 05-Jun-2022
#  Last update:  05-Jun-2022
#
################################################################

import sys, os
import numpy as np
import matplotlib.pylab as plt
import pickle

def Plot(x,c,m,e,alg,outdir):
    plt.plot(x,c,color='k')
    plt.errorbar(x,m,e,marker='o',color='k',linewidth=1)
    plt.xlabel("Product Order")
    plt.ylabel("Cost")
    plt.tight_layout(pad=0,w_pad=0,h_pad=0)
    plt.savefig(outdir+"/"+alg+".png", dpi=300)
    plt.savefig(outdir+"/"+alg+".eps", dpi=300)
    plt.close()

if (len(sys.argv) == 1):
    print()
    print("product_order <src> <outdir>")
    print()
    print("  <src>    - directory of go_store outputs")
    print("  <outdir> - output directory")
    print()
    exit(0)

aname = sys.argv[1]
outdir = sys.argv[2]

p = pickle.load(open("products.pkl","rb"))
cost = p[2][::-1]  #  expect costs, in order to be like this

algs = ["bare","de","ga","gwo","jaya","pso","ro"]

x = np.arange(24)
m = np.zeros((len(algs),24))
e = np.zeros((len(algs),24))

for z,alg in enumerate(algs):
    lines = [i[:-1] for i in open(aname+("/%s.txt" % alg))]
    order = np.zeros((10,24))
    k = 0
    i = 0
    while (k < len(lines)):
        if (lines[k].find("Product order") != -1):
            k += 1
            v = []
            for j in range(24):
                v.append(float(lines[k+j].split("$")[-1][:-1]))
            k += 24
            order[i,:] = np.array(v)
            i += 1
        else:
            k += 1
    m[z,:] = order.mean(axis=0)
    e[z,:] = order.std(ddof=1,axis=0) / np.sqrt(10)

os.system("rm -rf %s; mkdir %s" % (outdir,outdir))

Plot(x,cost,m[0],e[0],"bare",outdir)
Plot(x,cost,m[1],e[1],"de",outdir)
Plot(x,cost,m[2],e[2],"ga",outdir)
Plot(x,cost,m[3],e[3],"gwo",outdir)
Plot(x,cost,m[4],e[4],"jaya",outdir)
Plot(x,cost,m[5],e[5],"pso",outdir)
Plot(x,cost,m[6],e[6],"ro",outdir)

