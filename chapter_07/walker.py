#
#  file: walker.py
#
#  RTK, 27-Jun-2022
#  Last update:  27-Jun-2022
#
################################################################

import sys
import numpy as np
from matplotlib import cm
from PIL import Image
from RE import *


################################################################
#  Walk
#
def Walk(steps, cname, mode):
    """Do a random walk for the given number of steps and 
       color map"""

    try:
        cmap = cm.get_cmap(cname)
    except:
        cmap = cm.get_cmap("inferno")

    if (mode == 8):
        offset = [[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]]
    else:
        offset = [[0,-1],[1,0],[0,1],[-1,0]]
    X = [0]
    Y = [0]
    C = [cmap(0)]
    for i in range(steps):
        m = rng.random()
        X.append(X[-1] + offset[m][0])
        Y.append(Y[-1] + offset[m][1])
        c = cmap(int(256*i/steps))
        C.append((c[0],c[1],c[2]))

    return X,Y,C


################################################################
#  CreateOutputImage
#
def CreateOutputImage(X,Y,C, background):
    """Take the walk points and create the output image"""

    x = np.array(X)
    y = np.array(Y)
    xmin = x.min(); xmax = x.max()
    dx = xmax - xmin
    ymin = y.min(); ymax = y.max()
    dy = ymax - ymin
    img = np.zeros((dy,dx,4), dtype="uint8")
    
    if (background is not None) and (background != "none"):
        try:
            r = int(background[:2],16)
            g = int(background[2:4],16)
            b = int(background[4:],16)
            a = 255
        except:
            r,g,b,a = 0,0,0,0
    else:
        r,g,b,a = 0,0,0,0

    img[:,:,0] = r
    img[:,:,1] = g
    img[:,:,2] = b
    img[:,:,3] = a

    for i in range(len(x)):
        xx = int((dx-1)*(x[i] - xmin) / dx)
        yy = int((dy-1)*(y[i] - ymin) / dy)
        c = C[i]
        img[yy,xx,0] = int(255*c[0])
        img[yy,xx,1] = int(255*c[1])
        img[yy,xx,2] = int(255*c[2])
        img[yy,xx,3] = 255

    return img


################################################################

if (len(sys.argv) == 1):
    print()
    print("walker 4|8 <steps> <cmap> <background> <orientation> <output> [<kind>|<kind> <seed>]")
    print()
    print(" 4|8           - 4 or 8 connected")
    print("  <steps>      - number of steps")
    print("  <cmaps>      - color map names w/commas")
    print("  <background> - background color, 'none' transparent (hex rgb)")
    print("  <orientation>- landscape|portrait")
    print("  <output>     - output image name")
    print("  <kind>       - randomness source")
    print("  <seed>       - random seed")
    print()
    exit(0)

mode = 8 if (sys.argv[1] == "8") else 4
steps= int(sys.argv[2])
cnames = sys.argv[3].split(",")
background = sys.argv[4].lower()
orient = sys.argv[5].lower()
oname= sys.argv[6]

#  Use global randomness source
if (len(sys.argv) == 8):
    kind = sys.argv[7]
    rng = RE(kind=kind, mode="int", low=0, high=mode)
elif (len(sys.argv) == 9):
    kind = sys.argv[7]
    seed = int(sys.argv[8])
    rng = RE(kind=kind, mode="int", low=0, high=mode, seed=seed)
else:
    rng = RE(mode="int", low=0, high=mode)

X = []
Y = []
C = []

for cname in cnames:
    x,y,c = Walk(steps,cname,mode)
    X = X + x
    Y = Y + y
    C = C + c

img = CreateOutputImage(X,Y,C,background)
rows, cols, _ = img.shape

if (orient == "portrait"):
    if (rows < cols):
        img = img.transpose([1,0,2])
else:
    if (rows > cols):
        img = img.transpose([1,0,2])

Image.fromarray(img).save(oname)

