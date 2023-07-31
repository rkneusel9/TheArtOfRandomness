#
#  file:  birthday_true.py
#
#  Simulate the birthday paradox using the true distribution
#  of birthdays from insurance data
#
#  RTK, 01-May-2022
#  Last update:  01-May-2022
#
################################################################

import sys
import numpy as np
from RE import *
from fldr import *
from fldrf import *

#  Use real histogram of birthdays
#X = fldr_preprocess_float_c(list(np.loadtxt("bdata.txt")))
X = fldr_preprocess_float_c(list(np.loadtxt("bday.csv")))

def Simulate(M):
    """Simulate picking M people and at least two sharing a birthday"""

    matches = []
    for n in range(100_000):
        match = 0
        bdays = np.array([fldr_sample(X) for i in range(M)])
        for i in range(M-1):
            for j in range(i+1,M):
                if (bdays[i] == bdays[j]):
                    match += 1
        matches.append(match)

    #  Make a NumPy array and divide by two to eliminate double counting
    matches = np.array(matches)
    
    #  Return the histogram of matches
    return np.bincount(matches)

#
#  Main:
#
if (len(sys.argv) == 1):
    print()
    print("birthday <people> [<output>]")
    print()
    print("  <people>  -  Number of people in the room")
    print("  <output>  -  NumPy output file for matches histogram")
    print()
    exit(0)

#  Configure the randomness engine
people = int(sys.argv[1])
matches = Simulate(people)
prob = matches[1:].sum() / matches.sum()
print("%d people in the room, probability of at least 1 match = %0.6f" % (people, prob))
if (len(sys.argv) == 3):
    np.save(sys.argv[2], matches)

