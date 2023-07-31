#
#  file:  cs_image.py
#
#  Compressed sensing with images using LASSO
#
#  RTK, 17-Jul-2022
#  Last update:  20-Jul-2022
#
################################################################

import os
import sys
from sklearn.linear_model import Lasso
from scipy.fftpack import dct, idct
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from RE import *

try:
    from skimage.metrics import structural_similarity as ssim
    haveSSIM = True
except:
    haveSSIM = False

def CS(simg, mask, fraction, alpha, rng):
    """Apply CS"""

    row,col = simg.shape
    f = simg.ravel()
    N = len(f)
    k = np.where(mask != 0)[0]
    y = f[k]
    D = dct(np.eye(N))
    A = D[k, :]
    seed = int(10000000*rng.random())
    lasso = Lasso(alpha=alpha, max_iter=6000, tol=1e-4, random_state=seed)
    lasso.fit(A, y.reshape((len(k),)))
    r = idct(lasso.coef_.reshape((N, 1)), axis=0)
    r = (r - r.min()) / (r.max() - r.min())
    oimg = (255*r).astype("uint8").reshape((row,col))
    return oimg


#
# main
#
if (len(sys.argv) == 1):
    print()
    print("cs_image <image> <output> <fraction> <alpha> [ <kind> | <kind> <seed> ]")
    print()
    print("  <image>    - source image (RGB or grayscale)")
    print("  <output>   - output directory (overwrittten)")
    print("  <fraction> - fraction of image to sample")
    print("  <alpha>    - L1 lambda coefficient")
    print("  <kind>     - randomness source")
    print("  <seed>     - seed value")
    print()
    exit(0)

sname = sys.argv[1]
outdir = sys.argv[2]
fraction = float(sys.argv[3])
alpha = float(sys.argv[4])

if (len(sys.argv) == 7):
    if (sys.argv[5] == "quasi"):
        rng = RE(kind=sys.argv[5], base=int(sys.argv[6]))
    else:
        rng = RE(kind=sys.argv[5], seed=int(sys.argv[6]))
elif (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[5], base=3)
else:
    rng = RE()

simg = np.array(Image.open(sname).convert("RGB"))
grayscale = False
if (np.array_equal(simg[:,:,0],simg[:,:,1])):
    grayscale = True

#  Define the mask of randomly selected pixels
row, col, _ = simg.shape
mask = np.zeros(row*col, dtype="uint8")
M = int(fraction*row*col)
k = np.argsort(rng.random(row*col))[:M]
mask[k] = 1

#  Apply CS
if (grayscale):
    oimg = CS(simg[:,:,0], mask, fraction, alpha, rng)
else:
    oimg = np.zeros(simg.shape, dtype="uint8")
    for i in range(3):
        oimg[:,:,i] = CS(simg[:,:,i], mask, fraction, alpha, rng)

os.system("rm -rf %s; mkdir %s" % (outdir,outdir))
sn = os.path.basename(sname)
if (grayscale):
    Image.fromarray(simg[:,:,0]).save("%s/%s" % (outdir,sn))
else:
    Image.fromarray(simg).save("%s/%s" % (outdir,sn))
Image.fromarray(oimg).save("%s/output.png" % outdir)
Image.fromarray(255*mask.reshape((row,col))).save("%s/mask.png" % outdir)

with open("%s/params.txt" % outdir,"w") as f:
    f.write("fraction %0.4f\n" % fraction)
    f.write("alpha    %0.8f\n" % alpha)
    if (haveSSIM) and (grayscale):
        f.write("mssim    %0.8f\n" % ssim(oimg,simg[:,:,0]))

