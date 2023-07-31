#
#  file:  random_sounds.py
#
#  Play random samples
#
#  RTK, 28-Jun-2022
#  Last update:  28-Jun-2022
#
################################################################

import sys
import numpy as np
from scipy.io.wavfile import write as wavwrite

def WriteOutputWav(samples, name):
    s = (samples - samples.min()) / (samples.max() - samples.min())  # [0,1]
    s = (-1.0 + 2.0*s).astype("float32")     # [-1,1]
    wavwrite(name, rate, s)


if (len(sys.argv) == 1):
    print()
    print("random_sounds <dur> <output>")
    print()
    print("  <dur>     - duration (sec)")
    print("  <output>  - output .wav file")
    print()
    exit(0)

rate = 22050 # Hz
duration = float(sys.argv[1])
oname = sys.argv[2]

nsamples = int(duration * rate)
samples = -1.0 + 2.0*np.random.random(nsamples)

WriteOutputWav(samples, oname)

