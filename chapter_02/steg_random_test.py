#
#  file:  steg_random_test.py
#
#  RTK, 10-Apr-2022
#  Last update:  10-Apr-2022
#
################################################################

from RE import *
import numpy as np
import matplotlib.pylab as plt
import os

def ParseEntOutput():
    """Read and parse /tmp/ent"""
    s = open("/tmp/ent").readlines()
    return float(s[-2].split()[6])


#  Generate a temporary pool file
os.system("python3 make_random.py 10000000 none /tmp/pool urandom")

#  Embed ever larger collections of A's
N = [1000,5000,10000,50000,100000,500000,1000000]
os.system("ent /tmp/pool >/tmp/ent")
pi = [ParseEntOutput()]
for n in N:
    v = 65*np.ones(n, dtype="uint8")
    v.tofile("/tmp/source")
    os.system("python3 steg_random.py 3141592 /tmp/source /tmp/dest /tmp/pool")
    os.system("ent /tmp/dest >/tmp/ent")
    pi.append(ParseEntOutput())
N = [0] + N

#  Plot
N = np.array(N)
pi = np.array(pi)
d = np.zeros((len(N),2))
d[:,0] = N
d[:,1] = pi
np.save("steg_random_test_results.npy", d)
plt.plot(N, pi, marker='o', color='k')
plt.plot([N[0],N[-1]], [np.pi, np.pi], color='k', linestyle='dashed')
plt.xlabel("File size")
plt.ylabel("Estimate of $\pi$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("steg_random_test_plot.png", dpi=300)
plt.savefig("steg_random_test_plot.eps", dpi=300)

#  Clean up
os.system("rm /tmp/pool")
os.system("rm /tmp/source")
os.system("rm /tmp/dest")
os.system("rm /tmp/ent")

