#
#  file:  mcmc.py
#
#  Markov Chain Monte Carlo using randon walk Metropolis-Hastings
#
#  RTK, 16-Oct-2022
#  Last update:  27-Oct-2022
#
################################################################

from RE import *
import numpy as np
import time
import os
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


################################################################
#  MH
#
def MH(func, nsamples, sigma=1, q=1, burn=1000, limits=None):
    """Use 1D Metropolis-Hastings to sample func"""

    samples = [q]
    while (len(samples) < (burn+nsamples)):
        p = normal(q, sigma)
        if (limits is not None):
            lo,hi = limits
            if (p <= lo) or (p >= hi):
                p = q
        x = p; num = eval(func)
        x = q; den = eval(func)
        if (rng.random() < num/den):
            q = p
        samples.append(q)

    samples = np.array(samples)
    return samples[burn:], samples[:burn]


#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("mcmc <N> <func> <limits> <q> <sigma> <burn> <outdir> yes|no [<kind> | <kind> <seed>]")
    print()
    print("  <N>         - number of samples")
    print("  <func>      - function to sample from (e.g. 2*x**2+3)")
    print("  <limits>    - limits for samples (lo_hi, -18_18) or 'none'")
    print("  <q>         - initial sample (e.g. 0)")
    print("  <sigma>     - proposal distribution sigma (e.g. 1)")
    print("  <burn>      - initial samples to throw away (e.g. N//4)")
    print("  <outdir>    - output directory name (overwritten)")
    print("  yes|no      - show or don't show the initial proposal distribution")
    print("  <kind>      - randomness source")
    print("  <seed>      - seed")
    print()
    exit(0)

N = int(sys.argv[1])
func = sys.argv[2]
limits = sys.argv[3]
q0 = float(sys.argv[4])
sigma = float(sys.argv[5])
burn = int(sys.argv[6])
oname = sys.argv[7]
showProposal = True if (sys.argv[8].lower() == "yes") else False

#  define the global RE instance
if (len(sys.argv) == 11):
    rng = RE(kind=sys.argv[9], seed=int(sys.argv[10]))
elif (len(sys.argv) == 10):
    rng = RE(kind=sys.argv[9])
else:
    rng = RE()

#  set the limits
if (limits.lower() != "none"):
    lo,hi = [float(i) for i in limits.split("_")]
    limits = (lo,hi)
else:
    limits = None

#  Gather the samples
s = time.time()
samples, burn = MH(func, N, q=q0, sigma=sigma, burn=burn, limits=limits)
e = time.time() - s

#  Create output
os.system("rm -rf %s; mkdir %s" % (oname,oname))

#  plot the histogram of the samples
h,x = np.histogram(samples, bins=60)
h = h / h.sum()  # make a distribution
h = h / h.max()  # scale [0,1] to compare with func
x = 0.5*(x[1:] + x[:-1])
plt.bar(x,h, width=0.9*(x[1]-x[0]), color='k', fill=False, linewidth=0.7)

#  plot the actual distribution
if (limits is None):
    x = np.linspace(x.min(),x.max(),1000)
else:
    x = np.linspace(limits[0], limits[1], 1000)
y = eval(func)
y = y / y.max()  # scale [0,1]
plt.plot(x,y,color='k')

#  plot the proposal distribution for q0
if (showProposal):
    t = np.linspace(q0-4*sigma,q0+4*sigma,1000)
    z = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-q0)/sigma)**2)
    z = z / z.max()  # force [0,1]
    plt.plot(t,z, linewidth=0.7, linestyle='dashed', color='k')

plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+"/histogram.png", dpi=300)
plt.savefig(oname+"/histogram.eps", dpi=300)
plt.close()

#  Now the trace plot
inc = len(samples)//300
if (inc < 1):
    inc = 1
x = np.arange(len(samples))
plt.plot(x[::inc], samples[::inc], color='k', linewidth=0.7)
plt.xlabel("Sample number")
plt.ylabel("$x$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+"/trace_plot.png", dpi=300)
plt.savefig(oname+"/trace_plot.eps", dpi=300)
plt.close()

#  And burn plot
inc = len(burn)//300
if (inc < 1):
    inc = 1
x = np.arange(len(burn))
plt.plot(x[::inc], burn[::inc], color='k', linewidth=0.7)
plt.xlabel("Burn number")
plt.ylabel("$x$")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+"/burn_plot.png", dpi=300)
plt.savefig(oname+"/burn_plot.eps", dpi=300)
plt.close()

#  Report the number of trials to get the desired samples
st = "%d samples in %0.4f s" % (N, e)
with open(oname+"/results.txt","w") as f:
    f.write(st+"\n")
print(st)

#  Store the samples
np.save(oname+"/samples.npy", samples)

