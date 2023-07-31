#
#  file:  birthday.py
#
#  Simulate the birthday paradox
#
#  RTK, 16-Apr-2022
#  Last update:  01-May-2022
#
################################################################

import sys
import numpy as np
from RE import *

def Simulate(rng, M):
    """Simulate picking M people and at least two sharing a birthday"""

    matches = []
    for n in range(100_000):
        match = 0
        bdays = rng.random(M)
        for i in range(M-1):
            for j in range(i+1,M):
                if (bdays[i] == bdays[j]):
                    match += 1
        matches.append(match)

    #  Make a NumPy array
    matches = np.array(matches)
    
    #  Return the histogram of matches
    return np.bincount(matches)

#
#  Main:
#
if (len(sys.argv) == 1):
    print()
    print("birthday <people> pcg64|mt19937|minstd|urandom|rdrand|<filename> [<output>]")
    print()
    print("  <people>  -  Number of people in the room")
    print("  <output>  -  NumPy output file for matches histogram")
    print()
    exit(0)

#  Configure the randomness engine
people = int(sys.argv[1])
rng = RE(kind=sys.argv[2], low=0, high=365, mode="int")
matches = Simulate(rng, people)
prob = matches[1:].sum() / matches.sum()
print("%d people in the room, probability of at least 1 match = %0.6f" % (people, prob))
if (len(sys.argv) == 4):
    np.save(sys.argv[3], matches)

