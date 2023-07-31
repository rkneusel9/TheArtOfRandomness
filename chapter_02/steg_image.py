#
#  file:  steg_image.py
#
#  Embed a file in an image
#
#  RTK, 01-Apr-2022
#  Last update:  01-Apr-2022
#
################################################################

import numpy as np
from PIL import Image
import sys
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
#  Encode
#
def Encode(mfile, sfile, dfile):
    """Embed a message in an image"""
    
    msg = MakeBit(np.fromfile(mfile, dtype="uint8"))
    simg = np.array(Image.open(sfile).convert("RGB"))
    simg[np.where(simg > 253)] = 253
    row, col, channel = simg.shape
    simg = simg.ravel()
    if (len(msg) > len(simg)):
        print("Message file too long")
        exit(1)
    rng = RE(kind="mt19937")
    p = np.arange(len(simg))
    np.random.shuffle(p)
    p = p[:len(msg)]
    p.sort()
    for i in range(len(p)):
        simg[p[i]] += msg[i]+1
    simg = simg.reshape((row, col, channel))
    Image.fromarray(simg).save(dfile)


################################################################
#  Decode
#
def Decode(dfile, sfile, mfile):
    """Extract a message"""

    dimg = np.array(Image.open(dfile)).ravel()
    simg = np.array(Image.open(sfile).convert("RGB")).ravel()
    simg[np.where(simg > 253)] = 253
    i = np.where(dimg != simg)
    d = dimg[i] - simg[i] - 1
    b = MakeByte(d.astype("uint8"))
    b.tofile(mfile)


#
#  Main:
#
if (len(sys.argv) == 1):
    print()
    print("steg_image encode <message> <source> <dest> | decode <dest> <source> <message>")
    print("  <message> - message to embed")
    print("  <source>  - reference image")
    print("  <dest>    - output image")
    print()
    exit(0)

if (sys.argv[1] == "encode"):
    mfile = sys.argv[2]
    sfile = sys.argv[3]
    dfile = sys.argv[4]
    Encode(mfile, sfile, dfile)
else:
    dfile = sys.argv[2]
    sfile = sys.argv[3]
    mfile = sys.argv[4]
    Decode(dfile, sfile, mfile)

