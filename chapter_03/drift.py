#
#  file:  drift.py
#
#  A simple example of genetic drift
#
#  RTK, 03-May-2022
#  Last update:  03-May-2022
#
################################################################

import numpy as np
from RE import *

rng = RE(low=0, high=10, mode="int")
pop = rng.random(10_000)

sub = np.zeros((20,50), dtype="uint8")
for i in range(20):
    idx = np.argsort(np.random.random(len(pop)))
    sub[i,:] = pop[idx][:50]

print()
print("Population mean = %0.6f" % pop.mean())
print()
print("Sub-population means:")
for i in range(20):
    print("    %0.2f" % sub[i].mean())

