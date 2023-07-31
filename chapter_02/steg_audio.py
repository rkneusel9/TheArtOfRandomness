#
#  file:  steg_audio.py
#
#  Embed a file in audio
#
#  RTK, 01-Apr-2022
#  Last update:  11-Apr-2022
#
################################################################

import sys
import numpy as np
from RE import *
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite


################################################################
#  MakeBit
#
def MakeBit(byt):
    """Convert an array of bytes to bits"""

    b = np.zeros(8*len(byt), dtype="uint8")
    k = 0
    for v in byt:
        s = format(v, "08b")
        for c in s:
            b[k] = int(c)
            k += 1
    return b


################################################################
#  MakeByte
#
def MakeByte(b):
    """Convert array of bits to bytes"""

    n = len(b)//8
    t = b.reshape((n,8))
    byt = np.zeros(n, dtype="uint8")
    for i in range(n):
        v = (t[i] * np.array([128,64,32,16,8,4,2,1])).sum()
        byt[i] = v
    return byt


################################################################
#  MessageLength
#
def MessageLength(bits):
    """Convert bits to uint32 message length"""

    b = MakeByte(np.array(bits))
    n = 256**3*b[0] + 256**2*b[1] + 256*b[2] + b[3]
    return n


################################################################
#  Decode
#
def Decode(key, sfile, dfile):
    """Extract a hidden message"""
    
    #  Load the WAV file 
    sample_rate, wav = wavread(sfile)
    if (wav.ndim == 2):
        samples = wav[:,0].astype("uint16")
    else:
        samples = wav.astype("uint16")

    #  First four bytes are the file length
    rng = RE(mode="int", low=1, high=5, seed=key)
    step = rng.random(32)
    bits = []
    idx = [step[0]]
    for i in range(1, len(step)):
        idx.append(idx[-1]+step[i])
    for i in range(len(idx)):
        bits.append(samples[idx[i]] % 2)
    n = MessageLength(bits)

    #  Get the position of each bit of the message
    offset = idx[-1]
    step = rng.random(8*n)
    idx = [offset + step[0]]
    bits = []
    for i in range(1, len(step)):
        idx.append(idx[-1]+step[i])
    for i in range(len(idx)):
        bits.append(samples[idx[i]] % 2)

    #  Convert to bytes and write to the output file
    msg = MakeByte(np.array(bits))
    msg.tofile(dfile)


################################################################
#  Encode
#
def Encode(key, sfile, dfile, pfile):
    """Hide a message"""
    
    #  Load message file and prefix length
    src = np.fromfile(sfile, dtype="uint8")
    s = format(len(src), "08x")
    b3 = int(s[0:2],16);  b2 = int(s[2:4],16)
    b1 = int(s[4:6],16);  b0 = int(s[6:8],16)
    src = MakeBit(np.hstack(([b3,b2,b1,b0],src)))

    #  Read the WAV file and make byte
    sample_rate, wav = wavread(pfile)
    if (wav.ndim == 2):
        samples = wav[:,0].astype("uint16")
    else:
        samples = wav.astype("uint16")

    #  Do we have enough samples?
    if (len(src) > len(samples)):
        print("The input WAV file is too short")
        exit(1)
    else:
        print("Using %d samples to store the file" % len(src))

    #  Select a random, but ordered, subset of the samples,
    #  one for each bit of the message.
    step = RE(seed=key, mode="int", low=1, high=5).random(len(src))
    idx = [step[0]]
    for i in range(1, len(src)):
        idx.append(idx[-1]+step[i])
    
    #  If the sound file is too small, complain and quit
    if (len(samples) <= idx[-1]):
        print("Audio file too short")
        exit(1)

    #  Store the bits in the least-significant position (bit 0)
    for i in range(len(src)):
        if (src[i] == 0):
            if ((samples[idx[i]] % 2) == 1):
                samples[idx[i]] -= 1
        else:
            if ((samples[idx[i]] % 2) == 0):
                samples[idx[i]] += 1

    #  And dump the samples to disk
    out = samples.astype("int16")
    if (wav.ndim == 2):
        wav[:,0] = out
        wavwrite(dfile, sample_rate, wav)
    else:
        wavwrite(dfile, sample_rate, out)


#
#  Main:
#
if (len(sys.argv) == 1):
    print()
    print("steg_audio <key> <source> <dest> [<pool>]")
    print()
    print("<key>    - pseudorandom generator key")
    print("<source> - source file to encode or decode")
    print("<dest>   - output file name")
    print("<pool>   - if present, random pool ==> encode")
    print("           otherwise, no pool ==> decode")
    print()
    exit(0)

key = int(sys.argv[1])
sfile = sys.argv[2]
dfile = sys.argv[3]
pfile = sys.argv[4] if len(sys.argv) == 5 else ""

if (pfile == ""):
    #  decode
    Decode(key, sfile, dfile)
else:
    #  encode
    Encode(key, sfile, dfile, pfile)

