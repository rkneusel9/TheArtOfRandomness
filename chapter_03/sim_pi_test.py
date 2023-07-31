#
#  file:  sim_pi_test.py
#
#  Test sim_pi.py
#
#  RTK, 15-Apr-2022
#  Last update:  25-Apr-2022
#
################################################################

import os
import numpy as np
import matplotlib.pylab as plt

if (not os.path.exists("sim_pi_test_results.npy")):
    M = 50        # number of simulations per randomness source
    N = 2_000_000  # number of simulated points

    #  Randomness sources to test
    sources = ["pcg64", "mt19937", "minstd", "urandom", "rdrand"]

    #  Run the tests
    results = []
    for src in sources:
        res = []
        for trial in range(M):
            os.system("python3 sim_pi.py %d %s >/tmp/pi" % (N,src))
            pi = float(open("/tmp/pi").read().split()[-1])
            res.append(pi)
        results.append(res)

    results = np.array(results)
    np.save("sim_pi_test_results.npy", results)

ax = plt.figure().add_subplot(111)
d = np.load("sim_pi_test_results.npy")
d = d.transpose()
plt.boxplot(d)
ax.set_xticklabels(['pcg64','mt19937','minstd','urandom','rdrand'])
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("sim_pi_test_plot.png", dpi=300)
plt.savefig("sim_pi_test_plot.eps", dpi=300)
plt.show()

