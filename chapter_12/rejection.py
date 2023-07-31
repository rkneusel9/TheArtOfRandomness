#
#  file:  rejection.py
#
#  Rejection sampling from a 1D continuous distribution
#
#  RTK, 14-Oct-2022
#  Last update:  23-Oct-2022
#
################################################################

from RE import *
import numpy as np
import time
import sys
import matplotlib.pylab as plt

#
#  N.B. the following assumg a global instance of RE() called 'rng'
#

################################################################
#  normal -- Box-Muller generated normal samples
#
def normal(mu=0, sigma=1):
    """Return a sample from a normal distribution"""

    if (normal.state):
        normal.state = False
        return sigma*normal.z2 + mu
    else:
        u1,u2 = rng.random(2)
        m = np.sqrt(-2.0*np.log(u1))
        z1 = m*np.cos(2*np.pi*u2)
        normal.z2 = m*np.sin(2*np.pi*u2)
        normal.state = True
        return sigma*z1 + mu
#  Set the initial state attribute
normal.state = False

def normal_function(mu,sigma,x):
    """Proposal as function of x, unnormalized"""
    return np.exp(-0.5*((x-mu)/sigma)**2)


################################################################
#  uniform
#
def uniform(a=0,b=1):
    """Wrapper to allow arbitrary range"""
    return a + (b-a)*rng.random()

def uniform_function(a,b,x):
    """Proposal as function of x, unnormalized"""
    if (type(x) is np.ndarray):
        return np.ones(len(x))
    else:
        return 1.0


#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("rejection <N> <proposal> <c> <func> <limits> <outdir> [<kind> | <kind> <seed>]")
    print()
    print("  <N>         - number of samples")
    print("  <proposal>  - uniform|normal_mu_sigma (e.g. normal_0_1)")
    print("  <c>         - proposal multiplier (e.g. 1)")
    print("  <func>      - function to sample from (e.g. 2*x**2+3)")
    print("  <limits>    - lo_hi limit on sampling range (e.g. -3_8.8)")
    print("  <outdir>    - output directory name (overwritten)")
    print("  <kind>      - randomness source")
    print("  <seed>      - seed")
    print()
    exit(0)

N = int(sys.argv[1])
ptype = sys.argv[2]
c = float(sys.argv[3])
func = sys.argv[4]
lo,hi = sys.argv[5].split("_")
lo,hi = float(lo),float(hi)
oname = sys.argv[6]

#  define the global RE instance
if (len(sys.argv) == 9):
    rng = RE(kind=sys.argv[7], seed=int(sys.argv[8]))
elif (len(sys.argv) == 8):
    rng = RE(kind=sys.argv[7])
else:
    rng = RE()

#  extract proposal distribution parameters
a,b = lo,hi
if (ptype != "uniform"):
    _, a, b = ptype.strip().split("_")
    a,b = float(a),float(b)

#  assign pg and g functions
if (ptype.find("normal") != -1):
    pg = normal
    g = normal_function
else:
    pg = uniform
    g = uniform_function

#  generate the samples
samples = []
trials = 0
k = 0
s = time.time()
while (k < N):
    #  sample the proposal distribution
    x = pg(a,b)

    #  calculate f(x) and c*g(x)
    F = eval(func)
    G = c*g(a,b,x)

    #  accept or reject
    if (rng.random()*G < F):
        samples.append(x)
        k += 1
    
    #  count this trial
    trials += 1

e = time.time() - s
samples = np.array(samples)

#  Create the output directory
os.system("rm -rf %s; mkdir %s" % (oname,oname))

#  Compare to distribution function -- scale both to [0,1]
H,X = np.histogram(samples, bins=60)
H = H / H.max()
X = 0.5*(X[1:] + X[:-1])
plt.bar(X,H, width=0.9*(X[1]-X[0]), color='k', fill=False, linewidth=0.7)
x = np.linspace(lo,hi,100)
y = eval(func)
y = y / y.max()
plt.plot(x,y, color='k')
plt.xlim((lo-0.05*lo,hi+0.05*hi))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+"/histogram.png", dpi=300)
plt.savefig(oname+"/histogram.eps", dpi=300)
plt.close()

#  Plot the distribution function and scaled proposal function
x = np.linspace(lo,hi,1000)
y = eval(func)
p = c*g(a,b,x)
plt.plot(x,y, linewidth=1.0, color='k')
plt.plot(x,p, linestyle='dashed', linewidth=0.7, color='k')
plt.xlim((lo-0.05*lo,hi+0.05*hi))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+"/plot.png", dpi=300)
plt.savefig(oname+"/plot.eps", dpi=300)
plt.close()

#  Report the number of trials to get the desired samples
st = "%d trials to get %d samples (%0.4f s)" % (trials, N, e)
with open(oname+"/results.txt","w") as f:
    f.write(st+"\n")
print(st)

#  Store the samples
np.save(oname+"/samples.npy", samples)



