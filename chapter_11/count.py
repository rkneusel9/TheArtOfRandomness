#
#  file:  count.py
#
#  Count by random sampling
#
#  RTK, 31-Aug-2022
#  Last update:  31-Aug-2022
#
################################################################

import sys
import matplotlib.pylab as plt
from RE import *

if (len(sys.argv) == 1):
    print()
    print("count <N> <reps> [<kind> | <kind> <seed>]")
    print()
    print("  <N>    - number of marbles in the bag")
    print("  <reps> - number of repetitions")
    print("  <kind> - randomness source")
    print("  <seed> - seed")
    print()
    exit(0)

N = int(sys.argv[1])
reps = int(sys.argv[2])

if (len(sys.argv) == 5):
    rng = RE(kind=sys.argv[3], mode="int", low=0, high=N, seed=int(sys.argv[4]))
elif (len(sys.argv) == 4):
    rng = RE(kind=sys.argv[3], mode="int", low=0, high=N)
else:
    rng = RE(mode="int", low=0, high=N)

counts = []
iterations = []

for i in range(reps):
    bag = np.zeros(N, dtype="uint8")
    k = 0
    while (True):
        n = rng.random()
        if (bag[n]):
            break
        bag[n] = 1
        k += 1
    M = int(0.8*(2*k**2)/np.pi)
    counts.append(M)
    iterations.append(k)

counts = np.array(counts)
iterations = np.array(iterations)

cm = int(counts.mean())
im = int(iterations.mean())
isum = iterations.sum()
print("N = %d, iterations %d, total %d" % (cm, im, isum))

plt.bar(range(1,reps+1), counts, width=0.8, color='k', hatch='/')
plt.plot([1, reps], [N,N], color='k')
plt.plot([1, reps], [counts.mean(), counts.mean()], color='k', linestyle='dashed')
plt.xlim((0,reps+1))
plt.xlabel("Repetition")
plt.ylabel("Estimate")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("count_plot.png", dpi=300)
plt.savefig("count_plot.eps", dpi=300)
plt.close()


