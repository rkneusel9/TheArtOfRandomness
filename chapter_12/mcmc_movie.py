#
#  file:  mcmc_movie.py
#
#  Markov Chain Monte Carlo using randon walk Metropolis-Hastings
#
#  RTK, 16-Oct-2022
#  Last update:  25-Oct-2022
#
################################################################

# python3 mcmc_movie.py 10000 "np.exp(-((x-5)/2)**2)+4*np.exp(-((x+5)/2)**2)" -18_18 0 3 1000 tmp yes 900 pcg64 66
# ffmpeg -framerate 10 -i frame_%04d.png -vf scale=1024:-1 ../../mcmc_movie.mp4

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
#  PlotFrame
#
def PlotFrame(func, q, sigma, limits, trials, oname, p, jump):
    """Plot a proposal frame"""

    t = np.linspace(q-4*sigma,q+4*sigma,1000)
    z = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-q)/sigma)**2)
    z = z / z.max()  # force [0,1]
    plt.plot(t,z, linewidth=0.7, linestyle='dashed', color='k')

    width = 1.4 if (jump) else 0.7
    plt.plot([p,p],[0,1], linewidth=width, linestyle='dashed', color='k')

    if (limits is None):
        x = np.linspace(t.min(),t.max(),1000)
    else:
        x = np.linspace(limits[0], limits[1], 1000)
    y = eval(func)
    y = y / y.max()  # scale [0,1]
    plt.plot(x,y,color='k')

    plt.xlim((x.min(),x.max()))
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(oname+("/frames/frame_%04d.png" % trials), dpi=300)
    plt.close()


################################################################
#  MH
#
def MH(func, nsamples, sigma=1, q=1, burn=1000, frames=200, limits=None, oname=""):
    """Use 1D Metropolis-Hastings to sample func"""

    samples = [q]
    trials = 0
    while (len(samples) < (burn+nsamples)):
        p = q + normal(0, sigma)
        if (limits is not None):
            lo,hi = limits
            if (p <= lo) or (p >= hi):
                p = q
        x = p; num = eval(func)
        x = q; den = eval(func)
        A = num / den
        rho = min(1,A)
        if (rng.random() < rho):
            q = p
        samples.append(q)
        if (trials >= burn) and (trials < (burn+frames)):
            PlotFrame(func, q, sigma, limits, trials-burn, oname, p, q==p)
        trials += 1
    return np.array(samples)[burn:], trials


#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("mcmc_movie <N> <func> <limits> <q> <sigma> <burn> <outdir> yes|no <frames> [<kind> | <kind> <seed>]")
    print()
    print("  <N>         - number of samples")
    print("  <func>      - function to sample from (e.g. 2*x**2+3)")
    print("  <limits>    - limits for samples (lo_hi, -18_18) or 'none'")
    print("  <q>         - initial sample (e.g. 0)")
    print("  <sigma>     - proposal distribution sigma (e.g. 1)")
    print("  <burn>      - initial samples to throw away (e.g. N//4)")
    print("  <outdir>    - output directory name (overwritten)")
    print("  yes|no      - show or don't show the initial proposal distribution")
    print("  <frames>    - number of frames to generate (e.g. 200)")
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
frames = int(sys.argv[9])

#  define the global RE instance
if (len(sys.argv) == 12):
    rng = RE(kind=sys.argv[10], seed=int(sys.argv[11]))
elif (len(sys.argv) == 11):
    rng = RE(kind=sys.argv[10])
else:
    rng = RE()

#  set the limits
if (limits.lower() != "none"):
    lo,hi = [float(i) for i in limits.split("_")]
    limits = (lo,hi)
else:
    limits = None

#  Create output
os.system("rm -rf %s; mkdir %s" % (oname,oname))
os.system("mkdir %s/frames" % oname)

#  Gather the samples
s = time.time()
samples, trials = MH(func, N, q=q0, sigma=sigma, burn=burn, frames=frames, limits=limits, oname=oname)
e = time.time() - s

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

#  Report the number of trials to get the desired samples
st = "%d samples in %0.4f s" % (N, e)
with open(oname+"/results.txt","w") as f:
    f.write(st+"\n")
print(st)

#  Store the samples
np.save(oname+"/samples.npy", samples)

