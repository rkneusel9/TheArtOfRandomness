#
#  file: cs_signal.py
#
#  1D CS example
#
#  RTK, 18-Jul-2022
#  Last update:  19-Jul-2022
#
################################################################

from scipy.io.wavfile import write as wavwrite
from sklearn.linear_model import Lasso
from scipy.fftpack import dct, idct
from RE import *
import matplotlib.pylab as plt
import numpy as np
import os
import sys

rate = 4096 # samples per second

def WriteOutputWav(samples, name):
    """Dump the samples, summed, to the given output file"""
    s = (samples - samples.min()) / (samples.max() - samples.min())  # [0,1]
    s = (-1.0 + 2.0*s).astype("float32")     # [-1,1]
    wavwrite(name, rate, s)

if (len(sys.argv) == 1):
    print()
    print("cs_signal <fraction> [ <kind> | <kind> <seed> ]")
    print()
    print("  <fraction> - fraction of samples to measure")
    print("  <kind>     - randomness source")
    print("  <seed>     - seed value")
    print()
    exit(0)

frac = float(sys.argv[1])

if (len(sys.argv) == 4):
    rng = RE(kind=sys.argv[2], seed = int(sys.argv[3]))
elif (len(sys.argv) == 3):
    rng = RE(kind=sys.argv[2])
else:
    rng = RE()

#  Generate a C chord
dur = 1.0   # seconds
f0,f1,f2 = 261.63, 329.63, 392.0  # Hz
samples  = np.sin(2*np.pi*np.arange(rate*dur)*f0/rate)
samples += np.sin(2*np.pi*np.arange(rate*dur)*f1/rate)
samples += np.sin(2*np.pi*np.arange(rate*dur)*f2/rate)

#  Measured samples
nsamp = int(frac*len(samples))
u = np.arange(0, len(samples), int(len(samples)/nsamp))  # uniform samples
bu = samples[u]
r = np.argsort(rng.random(len(samples)))[:nsamp]  # random samples
br = samples[r]

#  Show signal, uniform samples, and random samples
x = np.arange(len(samples))
plt.subplot(2,1,1)
plt.plot(x,samples, linewidth=0.5, color='k')
plt.plot(r,samples[r], fillstyle='none', linestyle='none', marker='o', color='k')
plt.xlim((0,500))
plt.ylabel('random')
plt.subplot(2,1,2)
plt.plot(x,samples, linewidth=0.5, color='k')
plt.plot(u,samples[u], fillstyle='none', linestyle='none', marker='s', color='k')
plt.xlim((0,500))
plt.xlabel('time')
plt.ylabel('uniform')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("cs_signal_samples.png", dpi=300)
plt.savefig("cs_signal_samples.eps", dpi=300)
plt.show()
plt.close()

#  Create the Theta matrices, one for each set of measurements
D = dct(np.eye(len(samples)))
U = D[u,:]
R = D[r,:]

#  Now fit each and plot the respective s vectors
lu = Lasso(alpha=0.01, max_iter=6000)
lu.fit(U, bu)
su = lu.coef_
lr = Lasso(alpha=0.01, max_iter=6000)
lr.fit(R, br)
sr = lr.coef_

plt.subplot(2,1,1)
plt.plot(su, linewidth=0.5, color='k')
plt.ylabel('uniform')
plt.subplot(2,1,2)
plt.plot(sr, linewidth=0.5, color='k')
plt.ylabel('random')
plt.xlabel('sparse vector')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("cs_signal_sparse.png", dpi=300)
plt.savefig("cs_signal_sparse.eps", dpi=300)
plt.show()
plt.close()

#  Reconstruct the signals and plot them
ru = idct(su.reshape((len(samples),1)), axis=0)
rr = idct(sr.reshape((len(samples),1)), axis=0)

x = np.arange(len(samples))
plt.subplot(3,1,1)
plt.plot(x,samples, linewidth=0.5, color='k')
plt.xlim((100,600))
plt.ylabel('original')
plt.subplot(3,1,2)
plt.plot(x,rr, linewidth=0.5, color='k')
plt.xlim((100,600))
plt.ylabel('random')
plt.subplot(3,1,3)
plt.plot(x,ru, linewidth=0.5, color='k')
plt.xlim((100,600))
plt.xlabel('time')
plt.ylabel('uniform')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("cs_signal_recon.png", dpi=300)
plt.savefig("cs_signal_recon.eps", dpi=300)
plt.show()
plt.close()

#  Write the reconstructed WAV files
WriteOutputWav(samples, "original.wav")
WriteOutputWav(rr, "recon_random.wav")
WriteOutputWav(ru, "recon_uniform.wav")

