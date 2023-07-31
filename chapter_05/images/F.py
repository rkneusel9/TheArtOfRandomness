#
#  file:  F.py
#
#  Apply parameters to an image.
#
#  RTK, 03-Jun-2022
#  Last update:  03-Jun-2022
#
################################################################

import sys
import numpy as np
from PIL import Image, ImageFilter

def ApplyEnhancement(g, a,b,c,k):
    """Enhance an image"""

    def stats(g,i,j):
        rlo = max(i-1,0); rhi = min(i+1,g.shape[0])
        clo = max(j-1,0); chi = min(j+1,g.shape[1])
        v = g[rlo:rhi,clo:chi].ravel()
        if len(v) < 3:
            return v[0],1.0
        return v.mean(), v.std(ddof=1)

    #  enhanced image
    rows,cols = g.shape
    dst = np.zeros((rows,cols))

    #  enhance
    G = g.mean()
    for i in range(rows):
        for j in range(cols):
            m,s = stats(g,i,j)
            dst[i,j] = ((k*G)/(s+b))*(g[i,j]-c*m)+m**a

    #  return enhanced image
    dmin = dst.min()
    dmax = dst.max()
    return (255*(dst - dmin) / (dmax - dmin)).astype("uint8")


def F(dst):
    r,c = dst.shape

    Is = Image.fromarray(dst).filter(ImageFilter.FIND_EDGES) 
    Is = np.array(Is)
    edgels = len(np.where(Is.ravel() > 20)[0])

    h = np.histogram(dst, bins=64)[0]
    p = h / h.sum()
    i = np.where(p != 0)[0]
    ent = -(p[i]*np.log2(p[i])).sum()
    
    F = np.log(np.log(Is.sum()))*(edgels/(r*c))*ent
    return F

#  main
if (len(sys.argv) == 1):
    print()
    print("F <src> <dst> <a> <b> <m> <k>")
    print()
    print("  <src>  - source grayscale image")
    print("  <dst>  -  output enhanced image")
    print("  <a>, <b>, <m>, <k> - enhancement parameters")
    print()
    exit(0)

src = sys.argv[1]
out = sys.argv[2]
a = float(sys.argv[3])
b = float(sys.argv[4])
m = float(sys.argv[5])
k = float(sys.argv[6])

img = np.array(Image.open(src).convert("L")) / 256.0
dst = ApplyEnhancement(img, a,b,m,k)
Image.fromarray(dst).save(out)
print("F = %0.8f" % F(dst))

