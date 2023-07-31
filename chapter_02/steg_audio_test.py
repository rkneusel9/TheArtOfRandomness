#
#  file:  steg_audio_test.py
#
#  Check a WAV file to see if it might have
#  a hidden file
#
#  RTK, 10-Apr-2022
#  Last update:  10-Apr-2022
#
################################################################

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.stats import chisquare

#  Read clean and altered WAV files
_, a = wavread("Fireflies.wav")
_, b = wavread("Attitude.wav")
_, c = wavread("Fun-Key.wav")
_, d = wavread("tmp.wav")

#  Get bit 0 of each sample for channel 0
abits = a[:,0].astype("uint16") % 2
bbits = b[:,0].astype("uint16") % 2
cbits = c[:,0].astype("uint16") % 2
dbits = d[:,0].astype("uint16") % 2

#  Display histogram and chisquare results
h = np.bincount(abits)
_, p = chisquare(h)
print("Fireflies:", h, p)
h = np.bincount(bbits)
_, p = chisquare(h)
print("Attitude :", h, p)
h = np.bincount(cbits)
_, p = chisquare(h)
print("Fun-Key  :", h, p)
h = np.bincount(dbits)
_, p = chisquare(h)
print("Encoded  :", h, p)

