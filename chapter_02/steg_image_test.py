#
#  file:  steg_image_test.py
#
#  RTK, 12-Apr-2022
#  Last update:  12-Apr-2022
#
################################################################

import os
import numpy as np
from RE import *

#  "A" and random up to the size of apples.png
M = 396000  # apples.png size when decompressed (bytes)
A = 65*np.ones(M, dtype="uint8")
B = RE(kind="rdrand", mode="byte").random(M)

#  Create an output image directory
os.system("rm -rf steg_image_test; mkdir steg_image_test")

#  Hide ever larger random and uniform data in apples.png
for p in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    n = int(p*M//8)
    print("Encoding using %d bits" % (8*n,))
    B[:n].tofile("/tmp/rand")
    A[:n].tofile("/tmp/A")
    cmd = "python3 steg_image.py encode /tmp/rand test_images/apples.png steg_image_test/apple_rand_%0.2f.png" % p
    os.system(cmd)
    cmd = "python3 steg_image.py encode /tmp/A test_images/apples.png steg_image_test/apple_A_%0.2f.png" % p
    os.system(cmd)

#  Repeat using violet.png
M = 786432  # violet.png decompressed (bytes)
A = 65*np.ones(M, dtype="uint8")
B = RE(kind="rdrand", mode="byte").random(M)

#  Hide ever larger random and uniform data in apples.png
for p in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    n = int(p*M//8)
    print("Encoding using %d bits" % (8*n,))
    B[:n].tofile("/tmp/rand")
    A[:n].tofile("/tmp/A")
    cmd = "python3 steg_image.py encode /tmp/rand test_images/violet.png steg_image_test/violet_rand_%0.2f.png" % p
    os.system(cmd)
    cmd = "python3 steg_image.py encode /tmp/A test_images/violet.png steg_image_test/violet_A_%0.2f.png" % p
    os.system(cmd)

