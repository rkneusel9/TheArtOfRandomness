#
#  file:  mark_recapture_range.py
#
#  Get a feel for the three estimators as population
#  and initial marked percentage change
#
#  RTK, 09-Sep-2022
#  Last update:  09-Sep-2022
#
################################################################

import matplotlib.pylab as plt
import numpy as np
from RE import *

rng = RE(seed=10141066)

def Simulate(npop, nmark, nreps=200):
    lincoln = []
    chapman = []
    bayes = []
    for j in range(nreps):
        pop = np.zeros(npop, dtype="uint8")
        idx = np.argsort(rng.random(npop))[:nmark]
        pop[idx] = 1
        K = nmark
        while (True):
            idx = np.argsort(rng.random(npop))[:K]
            k = pop[idx].sum()
            if (k > 2):
                break
            K += 5
        lincoln.append(nmark*K/k)
        chapman.append((nmark+1)*(K+1)/(k+1) - 1)
        bayes.append((nmark-1)*(K-1)/(k-2))
    a = np.median(np.array(lincoln))
    b = np.median(np.array(chapman))
    c = np.median(np.array(bayes))
    return a,b,c

X = [100,1000,2000,5000,7000,8500,10000]
Y = np.linspace(0.03,0.2,20)

styles = ['solid','dashed','dotted','dashdot',
    (0,(3,1,1,1)), (0,(3,5,1,5,1,5)), (0,(3,1,1,1,1,1))]

lincoln = np.zeros((len(X),len(Y)))
chapman = np.zeros((len(X),len(Y)))
bayes   = np.zeros((len(X),len(Y)))

for i in range(len(X)):
    for j in range(len(Y)):
        nmark = int(Y[j]*X[i])
        a,b,c = Simulate(X[i],nmark)
        lincoln[i,j] = (X[i]-a)/X[i]
        chapman[i,j] = (X[i]-b)/X[i]
        bayes[i,j]   = (X[i]-c)/X[i]

for i in range(len(X)):
    plt.plot(Y, lincoln[i,:], linestyle=styles[i], color='k', label="%d" % X[i])
plt.plot([Y[0],Y[-1]],[0,0], linewidth=0.6, color='k')
plt.xlabel("Marked fraction")
plt.ylabel("Deviation")
plt.legend(loc="upper right")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("mark_recapture_lincoln.eps", dpi=300)
plt.savefig("mark_recapture_lincoln.png", dpi=300)
plt.close()

for i in range(len(X)):
    plt.plot(Y, chapman[i,:], linestyle=styles[i], color='k', label="%d" % X[i])
plt.plot([Y[0],Y[-1]],[0,0], linewidth=0.6, color='k')
plt.xlabel("Marked fraction")
plt.ylabel("Deviation")
plt.legend(loc="upper right")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("mark_recapture_chapman.eps", dpi=300)
plt.savefig("mark_recapture_chapman.png", dpi=300)
plt.close()

for i in range(len(X)):
    plt.plot(Y, bayes[i,:], linestyle=styles[i], color='k', label="%d" % X[i])
plt.plot([Y[0],Y[-1]],[0,0], linewidth=0.6, color='k')
plt.xlabel("Marked fraction")
plt.ylabel("Deviation")
plt.legend(loc="lower right")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("mark_recapture_bayes.eps", dpi=300)
plt.savefig("mark_recapture_bayes.png", dpi=300)
plt.close()

