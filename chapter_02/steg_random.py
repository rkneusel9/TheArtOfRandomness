#
#  file:  steg_random.py
#
#  Hiding a file in random data
#
#  RTK, 30-Mar-2022
#  Last update:  30-Mar-2022
#
################################################################

import sys
import numpy as np
from RE import *

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
    
    #  Load the file with the hidden message as bits
    src = MakeBit(np.fromfile(sfile, dtype="uint8"))

    #  The first four bytes are the message length
    rng = RE(mode="int", low=1, high=16, seed=key)
    step = rng.random(32)
    bits = []
    idx = [step[0]]
    for i in range(1, len(step)):
        idx.append(idx[-1]+step[i])
    for i in range(len(idx)):
        bits.append(src[idx[i]])
    n = MessageLength(bits)

    #  Read that many bits continuing from last position
    offset = idx[-1]
    step = rng.random(8*n)
    idx = [offset + step[0]]
    bits = []
    for i in range(1, len(step)):
        idx.append(idx[-1]+step[i])
    for i in range(len(idx)):
        bits.append(src[idx[i]])
    dest = MakeByte(np.array(bits))
    dest.tofile(dfile)


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

    #  Get random steps based on supplied key
    step = RE(mode="int", low=1, high=16, seed=key).random(len(src))
    idx = [step[0]]
    for i in range(1, len(step)):
        idx.append(idx[-1]+step[i])
    
    pool = MakeBit(np.fromfile(pfile, dtype="uint8"))
    if (len(pool) <= idx[-1]):
        print("Pool file is too small")
        exit(1)

    #  Alter bits by steps to match source file
    for i in range(len(src)):
        pool[idx[i]] = src[i]

    #  Convert back to bytes and dump the output file
    dest = MakeByte(pool)
    dest.tofile(dfile)


#
#  Main:
#
if (len(sys.argv) == 1):
    print()
    print("steg_random <key> <source> <dest> [<pool>]")
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

