#
#  file:  permutation_sort_plot.py
#
#  Plot the mean number of iterations (and print search time)
#
#  RTK, 30-Aug-2022
#  Last update:  04-Sep-2022
#
################################################################

#  Runtimes:
#     2: 0.127855 +/- 0.002026
#     3: 0.128128 +/- 0.001737
#     4: 0.129859 +/- 0.002469
#     5: 0.131369 +/- 0.002483
#     6: 0.136637 +/- 0.003704
#     7: 0.172775 +/- 0.008236
#     8: 0.534369 +/- 0.081601
#     9: 1.987567 +/- 0.488691
#    10: 44.133984 +/- 10.929158

from RE import *
import numpy as np
import matplotlib.pylab as plt
import sys
import time

def Iterations():
    line = [i[:-1] for i in open("/tmp/xyzzy")][0]
    return int(line.split()[-2][1:])

rng = RE(kind="mt19937", mode="int", low=999, high=99999, seed=6510)
cmd = "python3 permutation_sort.py %d 0 pcg64 %d >/tmp/xyzzy"

x = np.arange(2,11, dtype="uint8")
y = []
for i in x:
    k = []
    t = []
    for j in range(10):
        s = time.time()
        os.system(cmd % (i, rng.random()))
        t.append(time.time()-s)
        k.append(Iterations())
    k = np.array(k)
    t = np.array(t)
    y.append(k.mean())
    print("%2d: %0.6f +/- %0.6f" % (i, t.mean(), t.std(ddof=1)/np.sqrt(len(t))))

y = np.array(y)
plt.plot(x, y, marker='o', linewidth=0.7, fillstyle='none', color='k')
plt.xticks(x, [str(i) for i in x])
plt.xlabel("items")
plt.ylabel("mean permutations (n=10)")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("permutation_sort_plot.png", dpi=300)
plt.savefig("permutation_sort_plot.eps", dpi=300)
plt.close()

