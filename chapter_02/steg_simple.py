#
#  file:  steg_simple.py
#
#  Hide a text message in text from another source
#  by using a fixed character offset
#
#  RTK, 01-Apr-2022
#  Last update:  01-Apr-2022
#
################################################################

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
def Encode(offset, sfile, pfile, dfile):
    """Encode a message file"""

    msg = ProcessText(open(sfile).read())
    pool= ProcessText(open(pfile).read())
    
    enc = []
    idx = 0
    for word in msg:
        for c in word:
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

    with open(dfile, "w") as f:
        f.write(" ".join(enc)+"\n")


################################################################
#  Decode
#
def Decode(offset, dfile, sfile):
    """Decode a message file"""

    enc = ProcessText(open(dfile).read())
    plain = ""
    for w in enc:
        plain += w[offset]
    with open(sfile, "w") as f:
        f.write(plain+"\n")


if (len(sys.argv) == 1):
    print()
    print("steg_simple encode <offset> <source> <pool> <dest> | decode <offset> <dest> <source>")
    print()
    print("  <offset> - character offset")
    print("  <pool>   - pool text")
    print("  <source> - embedding text")
    print("  <dest>   - embedded text")
    print()
    exit(0)

offset = int(sys.argv[2])
if (sys.argv[1] == "encode"):
    sfile = sys.argv[3]
    pfile = sys.argv[4]
    dfile = sys.argv[5]
    Encode(offset, sfile, pfile, dfile)
elif (sys.argv[1] == "decode"):
    dfile = sys.argv[3]
    sfile = sys.argv[4]
    Decode(offset, dfile, sfile)
else:
    print("Unknown option")

