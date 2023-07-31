#
#  file:  sine_walker.py
#
#  Random walks using sine waves
#
#  RTK, 28-Jun-2022
#  Last update:  28-Jun-2022
#
################################################################

import sys
from RE import *
from scipy.io.wavfile import write as wavwrite

#  Fix the sampling rate -- used globally
rate = 22050  # Hz

def WriteOutputWav(samples, name):
    """Dump the samples, summed, to the given output file"""

    s = (samples - samples.min()) / (samples.max() - samples.min())  # [0,1]
    s = (-1.0 + 2.0*s).astype("float32")     # [-1,1]
    wavwrite(name, rate, s)


if (len(sys.argv) == 1):
    print()
    print("sine_walker <dur> <walkers> <output> [<kind> | <kind> <seed0> ...]")
    print()
    print("  <dur>     - duration (sec)")
    print("  <walkers> - number of walkers")
    print("  <output>  - output .wav file")
    print("  <kind>    - randomness source")
    print("  <seed>... - seeds, one per walker")
    print()
    exit(0)

duration = float(sys.argv[1])
nwalkers = int(sys.argv[2])
oname = sys.argv[3]

if (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[4], seed=int(sys.argv[5]))
elif (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[4])
else:
    rng = RE()

nsamples = int(duration * rate)
samples = np.zeros(nsamples, dtype="float32")

#  step size
dur = 0.5
step_samp = int(dur * rate)
fstep = 5  # Hz

#  initial frequencies
freq = np.zeros(nwalkers, dtype="uint32")
freq[:] = (440 + 800*(rng.random(nwalkers)-0.5)).astype("uint32")

k = 0
while (k < nsamples):
    for i in range(nwalkers):
        r = rng.random()
        if (r < 0.33333):
            freq[i] += fstep
        elif (r < 0.66666):
            freq[i] -= fstep
        freq[i] = min(max(100,freq[i]),4000)
        amp = rng.random()
        if (i == 0):
            t = amp*np.sin(2*np.pi*np.arange(rate*dur)*freq[i]/rate)
        else:
            t += amp*np.sin(2*np.pi*np.arange(rate*dur)*freq[i]/rate)
    n = 1
    while (np.abs(t[-n]) > 1e-4):
        n += 1
    t = t[:-n]
    if ((k+len(t)) < nsamples):
        samples[k:(k+len(t))] = t
    k += len(t)

#  Clip to keep samples between the 10% and 90% percentiles
lo = np.quantile(samples, 0.1)
hi = np.quantile(samples, 0.9)
samples[np.where(samples <= lo)] = lo
samples[np.where(samples >= hi)] = hi

WriteOutputWav(samples, oname)

