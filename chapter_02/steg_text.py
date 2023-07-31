#
#  file:  steg_text.py
#
#  Hide a text message in text from another source
#
#  RTK, 01-Apr-2022
#  Last update:  01-Apr-2022
#
################################################################

from RE import *
import sys

################################################################
#  ProcessText
#
def ProcessText(s):
    """Process a text string"""

    s = s.upper().split()
    text = []
    for t in s:
        z = ""
        for c in t:
            if (c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                z += c
        text.append(z)
    return text


################################################################
#  Encode a message
#
def Encode(mfile, pfile, ofile):
    """Encode a message file"""

    msg = ProcessText(open(mfile).read())
    pool= ProcessText(open(pfile).read())
    key = RE(mode='int', low=10000, high=1000000).random()
    rng = RE(mode='int', low=1, high=5, seed=key)
    
    enc = []
    idx = 0
    for word in msg:
        for c in word:
            offset = rng.random()
            done = False
            while (not done) and (idx < len(pool)):
                if (len(pool[idx]) <= offset):
                    pass
                elif (pool[idx][offset] != c):
                    pass
                else:
                    enc.append(pool[idx])
                    done = True
                idx += 1

    with open(ofile, "w") as f:
        f.write(" ".join(enc)+"\n")
    print("Your secret key is %d" % key)


################################################################
#  Decode
#
def Decode(key, ofile, mfile):
    """Decode a message file"""

    enc = ProcessText(open(ofile).read())
    rng = RE(mode='int', low=1, high=5, seed=key)
    plain = ""

    for w in enc:
        plain += w[rng.random()]

    with open(mfile, "w") as f:
        f.write(plain+"\n")


if (len(sys.argv) == 1):
    print()
    print("steg_text <message> <pool> <output> | <key> <output> <message>")
    print()
    print("  <message> - plaintext message")
    print("  <pool>    - embedding text")
    print("  <output>  - embedded text")
    print("  <key>     - key for restoring plaintext")
    print()
    exit(0)

try:
    key = int(sys.argv[1])
except:
    mfile = sys.argv[1]
    pfile = sys.argv[2]
    ofile = sys.argv[3]
    Encode(mfile, pfile, ofile)
else:    
    ofile = sys.argv[2]
    mfile = sys.argv[3]
    Decode(key, ofile, mfile)

