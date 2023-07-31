#
#  file:  sine_walker.py
#
#  Random walks using sine waves
#
#  RTK, 28-Jun-2022
#  Last update:  11-Jul-2022
#
################################################################

import sys
from RE import *
from scipy.io.wavfile import write as wavwrite

frequencies = np.array([
146.83 ,  164.81 ,  174.61 ,  196.   ,
220.   ,  246.94 ,  261.63 ,  293.66 ,  329.63 ,  349.23 ,
392.   ,  440.   ,  493.88 ,  523.25 ,  587.33 ,  659.26 ,
698.46 ,  783.99 ,  880.   ,  987.77 , 1046.5  ])

#  Fix the sampling rate -- used globally
rate = 22050  # Hz

def WriteOutputWav(samples, name):
    """Dump the samples, summed, to the given output file"""
    s = (samples - samples.min()) / (samples.max() - samples.min())  # [0,1]
    s = (-1.0 + 2.0*s).astype("float32")     # [-1,1]
    wavwrite(name, rate, s)


if (len(sys.argv) == 1):
    print()
    print("sine_walker <dur> <walkers> <output> [<kind> | <kind> <seed>]")
    print()
    print("  <dur>     - duration (sec)")
    print("  <walkers> - number of walkers")
    print("  <output>  - output .wav file")
    print("  <kind>    - randomness source")
    print("  <seed>    - seed")
    print()
    exit(0)

duration = float(sys.argv[1])
nwalkers = int(sys.argv[2])
oname = sys.argv[3]

if (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[4])
elif (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[4], seed=int(sys.argv[5]))
else:
    rng = RE()

nsamples = int(duration * rate)
samples = np.zeros(nsamples, dtype="float32")

#  step duration
dur = 1.0

#  initial frequencies
freq = np.zeros(nwalkers, dtype="uint16")
freq[:] = np.array([int(len(frequencies)*rng.random()) for i in range(nwalkers)])

k = 0
while (k < nsamples):
    for i in range(nwalkers):
        r = rng.random()
        if (r < 0.33333):
            freq[i] += 1
        elif (r < 0.66666):
            freq[i] -= 1
        if (freq[i] < 0):
            freq[i] = 0
        if (freq[i] >= len(frequencies)):
            freq[i] = len(frequencies) - 1
        fr = frequencies[freq[i]]
        if (i == 0):
            t = np.sin(2*np.pi*np.arange(rate*dur)*fr/rate)
        else:
            t += np.sin(2*np.pi*np.arange(rate*dur)*fr/rate)
    n = 1
    while (np.abs(t[-n]) > 1e-4):
        n += 1
    t = t[:-n]
    if ((k+len(t)) < nsamples):
        samples[k:(k+len(t))] = t
    k += len(t)

WriteOutputWav(samples, oname)

