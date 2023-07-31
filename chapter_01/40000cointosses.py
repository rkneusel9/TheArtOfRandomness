#
#  file:  40000cointosses.py
#
#  process the 40000cointosses.csv
#  see: https://www.stat.berkeley.edu/~aldous/Real-World/coin_tosses.html
#
#  RTK, 15-Mar-2022
#  Last update: 16-Mar-2022
#
################################################################

import numpy as np
from scipy.stats import chisquare, ttest_ind

#  process the raw data
s = [i[:-1] for i in open("40000cointosses.csv")]
src = []
outcome = []
for t in s:
    if (t.find("H") != -1):
        src.append(1)
        if (t[13] == '1'):
            outcome.append(1)
        else:
            outcome.append(0)
    if (t.find("T") != -1):
        src.append(0)
        if (t[33] == '1'):
            outcome.append(1)
        else:
            outcome.append(0)

outcome = np.array(outcome, dtype="uint8")
src = np.array(src, dtype="uint8")

#
#  All tosses with src==1 by subject 1, all tosses with src==0 by
#  subject 2.  Subject 1 always started with the coin heads up, while
#  subject 2 started with the coin tails up.
#
heads = outcome[np.where(src==1)]
tails = outcome[np.where(src==0)]

#  Counts
print("Counts:")
b = np.bincount(outcome)
print("    overall: %5d heads, %5d tails" % (b[1],b[0]))
b = np.bincount(heads)
print("    heads  : %5d heads, %5d tails" % (b[1],b[0]))
b = np.bincount(tails)
print("    tails  : %5d heads, %5d tails" % (b[1],b[0]))
print()

#  Use chi square test to see if balanced
print("Chi square test:")
_, p = chisquare(np.bincount(outcome))
print("    overall: p=%0.8f" % p)
_, p = chisquare(np.bincount(heads))
print("    heads  : p=%0.8f" % p)
_, p = chisquare(np.bincount(tails))
print("    tails  : p=%0.8f" % p)

#  Are the two datasets from the same distribution?
print()
_, p = ttest_ind(heads, tails)
print("t-test: %0.8f" % p)
print()

#  Apply von Neumann "correction" to the Subject 1's flips
flips = []
k = 0
while (k < len(heads)-1):
    if (heads[k] != heads[k+1]):
        flips.append(heads[k]) 
    k += 2
b = np.bincount(flips)
_, p = chisquare(b)
print("Heads using von Neumann algorithm:")
print("    %4d heads, %4d tails, p=%0.8f" % (b[1], b[0], p))
print()

