#
#  file:  freivalds_plots.py
#
#  Plots related to Freivalds' algorithm
#
#  RTK, 31-Aug-2022
#  Last update: 31-Aug-2022
#
################################################################

import matplotlib.pylab as plt
import numpy as np
import os

def GetTime():
    t = [i[:-1] for i in open("/tmp/xyzzy")][0]
    return float(t.split()[0])

#  Validation time as function of matrix size
N = np.array([5,10,15,20,25,30,35,40])
naive = []
freivalds = []

for n in N:
    u = []
    v = []
    for k in range(5):
        os.system("python3 freivalds.py %d 0 1 >/tmp/xyzzy" % (n,))
        u.append(GetTime())
        os.system("python3 freivalds.py %d 1 1 >/tmp/xyzzy" % (n,))
        v.append(GetTime())
    naive.append(np.array(v).mean())
    freivalds.append(np.array(u).mean())

plt.plot(N, freivalds, marker='o', fillstyle='none', linewidth=0.7, color='k', label='Freivalds')
plt.plot(N, naive, marker='s', fillstyle='none', linewidth=0.7, color='k', label='Naive')
plt.xlabel('Matrix size')
plt.ylabel('Mean evaluation time (s)')
plt.legend(loc="upper left")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("freivalds_time_plot.png", dpi=300)
plt.savefig("freivalds_time_plot.eps", dpi=300)
plt.close()

plt.plot(N, freivalds, marker='o', fillstyle='none', linewidth=0.7, color='k', label='Freivalds')
plt.xlabel('Matrix size')
plt.ylabel('Mean evaluation time (s)')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("freivalds_only_time_plot.png", dpi=300)
plt.savefig("freivalds_only_time_plot.eps", dpi=300)
plt.close()

