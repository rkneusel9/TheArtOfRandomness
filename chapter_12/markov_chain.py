#
#  file:  markov_chain.py
#
#  Run a Markov chain to a stationary state
#
#  RTK, 26-Oct-2022
#  Last update:  25-Oct-2022
#
################################################################

from PIL import Image
import numpy as np
import matplotlib.pylab as plt
import os
import sys

#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("markov_chain <red> <green> <blue> <transition>")
    print()
    print("  <red> <green> <blue> -- initial distribution")
    print("  <transition> -- 3x3 transition matrix in NumPy form")
    print()
    exit(0)

#  initial distribution of red, green, and blue
r = float(sys.argv[1])
g = float(sys.argv[2])
b = float(sys.argv[3])
d = np.array([r/(r+g+b),g/(r+g+b),b/(r+g+b)])

#  normalize the transition matrix so each row sums to 1
transition = eval("np.array("+sys.argv[4]+",dtype='float64')")
transition[0,:] = transition[0,:] / transition[0,:].sum()
transition[1,:] = transition[1,:] / transition[1,:].sum()
transition[2,:] = transition[2,:] / transition[2,:].sum()

#  Build the chain until reaching a stationary state
eps = 1e-5
last = np.array([10,10,10])
chain = []

while (np.abs(d-last).sum() > eps):
    print(d)                    # current distribution
    chain.append(d)             # keep the current distribution
    last = d
    d = d @ transition

#  Turn the chain into an image using red, green, blue as the color
#  The image is 50 pixels high and 512 wide.  m is how many columns
#  each band occupies which depends on the number of links in the chain
m = int(512 / len(chain))
img = np.zeros((50,len(chain)*m,3), dtype="uint8")
k = 0
for c in chain:
    r,g,b = (255*c).astype("uint8")
    for j in range(5):
        img[:,k:(k+5),0] = r
        img[:,k:(k+5),1] = g
        img[:,k:(k+5),2] = b
        k += 5

im = Image.fromarray(img)
im.save("markov_chain.png")

#  end by printing the normalized transition matrix
print()
print(transition)

