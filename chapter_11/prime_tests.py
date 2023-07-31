#
#  file:  prime_tests.py
#
#  Compare the runtimes for brute force and Miller-Rabin for
#  the largest primes by number of digits
#
#  RTK, 12-Sep-2022
#  Last update:  12-Sep-2022
#
################################################################

import os
import numpy as np
import matplotlib.pylab as plt

def GetTime():
    t = [i[:-1] for i in open("/tmp/xyzzy")][0]
    return float(t.split()[-1][1:-1])

miller = []
brute = []

primes = [
    7, 97, 997, 9973, 99991, 999983, 9999991, 99999989,
    999999937, 9999999967, 99999999977, 999999999989,
    9999999999971, 99999999999973, 999999999999989]
digits = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for p in primes:
    v = []
    for i in range(5):
        os.system("python3 miller_rabin.py %d 1 >/tmp/xyzzy" % p)
        v.append(GetTime())
    miller.append(np.array(v).mean())

    v = []
    for i in range(5):
        os.system("python3 brute_primes.py %d >/tmp/xyzzy" % p)
        v.append(GetTime())
    brute.append(np.array(v).mean())

plt.plot(digits, miller, marker='o', fillstyle='none', linewidth=0.7, color='k', label='Miller-Rabin')
plt.plot(digits, brute, marker='^', fillstyle='none', linewidth=0.7, color='k', label='brute force')
plt.xticks(digits, [str(i) for i in digits])
plt.xlabel("Number of digits")
plt.ylabel("Runtime (s)")
plt.legend(loc="upper left")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("prime_tests.eps", dpi=300)
plt.savefig("prime_tests.png", dpi=300)
plt.close()

