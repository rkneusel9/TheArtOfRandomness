from Quicksort import *
import numpy as np
import matplotlib.pylab as plt
import time

#  Compare random arrays
M = 20
N = np.array([1000,3000,6000,9000,12000,15000,18000,21000,24000,27000,30000])
ran, non = [], []
for n in N:
    q, r = [], []
    for i in range(M):
        A = np.arange(n)
        np.random.shuffle(A)
        s = time.time()
        _ = Quicksort(A)
        q.append(time.time()-s)
        s = time.time()
        _ = QuicksortRandom(A)
        r.append(time.time()-s)
    ran.append(np.array(r).mean())
    non.append(np.array(q).mean())

#  Divide by the maximum to compare on similar y-axis range
ran = np.array(ran)
ran = ran / ran.max()
non = np.array(non)
non = non / non.max()
y = N*np.log(N)
y = y / y.max()

plt.plot(N, ran, marker='o', fillstyle='none', color='k', linewidth=0.7, label='QuicksortRandom')
plt.plot(N, non, marker='s', fillstyle='none', color='k', linewidth=0.7, label='Quicksort')
plt.plot(N, y, linewidth=0.7, color='k', linestyle='dashed')
plt.xlabel("$N$")
plt.ylabel("Scaled runtime")
plt.legend(loc='upper left')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("quicksort_tests_random.eps", dpi=300)
plt.savefig("quicksort_tests_random.png", dpi=300)
plt.close()

#  Now, pathological case of a sorted array (forward or reverse doesn't matter)
N = np.array([1000,3000,6000,9000,12000,15000,20000])
ran, non = [], []
for n in N:
    q, r = [], []
    for i in range(M):
        A = np.arange(n)
        s = time.time()
        _ = Quicksort(A)
        q.append(time.time()-s)
        s = time.time()
        _ = QuicksortRandom(A)
        r.append(time.time()-s)
    ran.append(np.array(r).mean())
    non.append(np.array(q).mean())

#  Divide by the maximum to compare on similar y-axis range
ran = np.array(ran)
ran = ran / ran.max()
non = np.array(non)
non = non / non.max()
y = N*np.log(N)
y = y / y.max()
z = N**2
z = z / z.max()

plt.plot(N, ran, marker='o', fillstyle='none', color='k', linewidth=0.7, label='QuicksortRandom')
plt.plot(N, non, marker='s', fillstyle='none', color='k', linewidth=0.7, label='Quicksort')
plt.plot(N, y, linewidth=0.7, color='k', linestyle='dashed')
plt.plot(N, z, linewidth=0.7, color='k', linestyle='dotted')
plt.xlabel("$N$")
plt.ylabel("Scaled runtime")
plt.legend(loc='upper left')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("quicksort_tests_pathological.eps", dpi=300)
plt.savefig("quicksort_tests_pathological.png", dpi=300)
plt.close()

