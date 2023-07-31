#
#  file:  process_images.py
#
#  Process color images channel by channel
#
#  RTK, 02-Jun-2022
#  Last update:  02-Jun-2022
#
################################################################

import numpy as np
import os
import sys
from PIL import Image

if (len(sys.argv) == 1):
    print()
    print("process_rgb_images <npart> <niter> <alg>")
    print()
    print("  <npart> - number of particles (e.g. 10)")
    print("  <niter> - number of iterations (e.g. 75)")
    print("  <alg>   - bare,de,ga,gwo,jaya,pso,ro")
    print()
    exit(0)

npart = int(sys.argv[1])
niter = int(sys.argv[2])
alg = sys.argv[3]

names = ["apples", "mandril", "peppers"]

for name in names:
    rgb = np.array(Image.open("original/"+name+".png").convert("RGB"))
    Image.fromarray(rgb[:,:,0]).save("/tmp/red.png")
    Image.fromarray(rgb[:,:,1]).save("/tmp/green.png")
    Image.fromarray(rgb[:,:,2]).save("/tmp/blue.png")
    os.system("python3 enhance.py /tmp/red.png %d %d %s mt19937 /tmp/red" % (npart,niter,alg))
    os.system("python3 enhance.py /tmp/green.png %d %d %s mt19937 /tmp/green" % (npart,niter,alg))
    os.system("python3 enhance.py /tmp/blue.png %d %d %s mt19937 /tmp/blue" % (npart,niter,alg))
    r = np.array(Image.open("/tmp/red/enhanced.png"))
    g = np.array(Image.open("/tmp/green/enhanced.png"))
    b = np.array(Image.open("/tmp/blue/enhanced.png"))
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b
    Image.fromarray(rgb).save("enhanced/"+name+".png")


