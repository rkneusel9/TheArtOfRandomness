#
#  file:  discrete_2d.py
#
#  RTK, 13-Oct-2022
#  Last update:  22-Oct-2022
#
################################################################

from RE import *
from fldr import fldr_preprocess_int, fldr_sample
from PIL import Image
import numpy as np
import matplotlib.pylab as plt
import time
import sys

#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("discrete_2d <image> <N> <outdir> [<kind> | <kind> <seed>]")
    print()
    print("  <image>  - input image (histogram source)")
    print("  <N>      - number of samples")
    print("  <outdir> - output directory (overwritten)")
    print("  <kind>   - randomness source")
    print("  <seed>   - seed")
    print()
    exit(0)

iname = sys.argv[1]
N = int(sys.argv[2])
oname = sys.argv[3]

if (len(sys.argv) == 6):
    rng = RE(kind=sys.argv[4], seed=int(sys.argv[5]))
elif (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[4])
else:
    rng = RE()

#  Scale the grayscale image then
#  unravel to use as the distribution
image = Image.open(iname).convert("L")
row, col = image.size
row //= 2
col //= 2
image = np.array(image.resize((row,col),Image.BILINEAR))
p = image.ravel()
probabilities = [int(t) for t in p]

#  Fast loaded dice roller
x = fldr_preprocess_int(probabilities)
z = np.array([fldr_sample(x) for i in range(N)])

#  Build 2D histogram with (x,y) samples extracted from z
x,y = np.unravel_index(z, (col,row))
im = np.zeros((col,row))
for i in range(len(x)):
    im[x[i],y[i]] += 1
im = im / im.max()

#  Output
os.system("rm -rf %s; mkdir %s" % (oname,oname))
Image.fromarray(image).save(oname+"/"+os.path.basename(iname))
Image.fromarray((255*im).astype("uint8")).save(oname+"/histogram2d.png")

