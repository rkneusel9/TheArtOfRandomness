#
#  file:  permutation_sort.py
#
#  Horribly inefficient permutation sort.  Run at Las Vegas
#  or Monte Carlo algorithm.
#
#  RTK, 30-Aug-2022
#  Last update:  30-Aug-2022
#
################################################################

from RE import *
import sys

################################################################
#  Score -- how sorted is it?
#
def Score(arg):
    """Return fraction of pairs out of place"""
    n = 0
    for i in range(len(arg)-1):
        if (arg[i] > arg[i+1]):
            n += 1
    return n / len(arg)


#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("permutation_sort <items> <limit> [<kind> | <kind> <seed>]")
    print()
    print("  <items> - number of items in the list")
    print("  <limit> - number of passes maximum (0=Las Vegas else Monte Carlo)")
    print("  <kind>  - randomness source")
    print("  <seed>  - seed value")
    print()
    exit(0)

N = int(sys.argv[1])
limit = int(sys.argv[2])

if (limit == 0):
    limit = 999999999999999999999

if (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[3], seed=int(sys.argv[4]))
elif (len(sys.argv) == 4):
    rng = RE(kind=sys.argv[3])
else:
    rng = RE()

#  Build the unsorted list
v = np.array([int(rng.random()*100) for i in range(N)], dtype="uint8")

#  Sort by shuffling
k = 0
score = Score(v)
while (score != 0) and (k < limit):
    k += 1
    i = np.argsort(rng.random(len(v)))
    s = Score(v[i])
    if (s < score):
        score = s
        v = v[i]

if (score == 0):
    print("sorted: ", end="")
else:
    print("partially: ", end="")

for i in range(len(v)):
    print("%d  " % v[i], end="")

if (score == 0):
    print("(%d iterations)" % k)
else:
    print("(score = %0.5f, %d iterations)" % (score, k))
print()


