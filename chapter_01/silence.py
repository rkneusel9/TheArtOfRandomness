#
#  file:  silence.py
#
#  Process a float32 .wav file of silence from the microphone
#  input with nothing connected.
#
#  RTK, 23-Mar-2022
#  Last update:  23-Mar-2022
#
################################################################

import sys
import numpy as np
from scipy.io.wavfile import read as wavread

def MakeBytes(A):
    t = A - A.mean()
    thresh = (t.max()-t.min())/100.0
    w = []
    for i in range(len(t)):
        if (np.abs(t[i]) < thresh):
            continue
        w.append(1 if t[i] > 0 else 0)
    b = []
    k = 0
    while (k < len(w)-1):
        if (w[k] != w[k+1]):
            b.append(w[k])
        k += 2
    n = len(b)//8
    c = np.array(b[:8*n]).reshape((n,8))
    z = []
    for i in range(n):
        t = (c[i] * np.array([128,64,32,16,8,4,2,1])).sum()
        z.append(t)
    return np.array(z).astype("uint8")

if (len(sys.argv) == 1):
    print()
    print("silence <source> <dest>")
    print()
    print("  <source> - float32 .wav file of silence from microphone input")
    print("  <dest>   - output file of bytes")
    print()
    exit(0)

s, d = wavread(sys.argv[1])
print("sampling rate: %d" % s)
n = len(d)//2
a = MakeBytes(d[:n])
b = MakeBytes(d[n:])
if (len(a) < len(b)):
    c = a[::-1] ^ b[:len(a)]
else:
    c = a[:len(b)] ^ b[::-1]
c.tofile(sys.argv[2])

