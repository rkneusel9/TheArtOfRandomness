#
#  file: process_results.py
#
#  Parse store runs
#
#  RTK, 03-Jun-2022
#  Last update: 03-Jun-2022
#
################################################################

import matplotlib.pylab as plt
import os, numpy as np
from scipy.stats import ttest_ind

os.system("rm -rf results; mkdir results")

milk = np.zeros((7,10))
candy = np.zeros((7,10))
revenue = np.zeros((7,10))

for k,alg in enumerate(["bare","de","ga","gwo","jaya","pso","ro"]):
    lines = [i[:-1] for i in open("output/"+alg.lower()+".txt")]
    r = []; m = []; c = []
    for line in lines:
        if (line.find("revenue") != -1):
            r.append(float(line.split()[3][1:]))
        if (line.find("milk rank") != -1):
            m.append(float(line.split()[-1]))
        if (line.find("candy rank") != -1):
            c.append(float(line.split()[-1]))
    milk[k,:] = np.array(m)
    candy[k,:] = np.array(c)
    revenue[k,:] = np.array(r)

np.save("results/milk_ranking.npy", milk)
np.save("results/candy_ranking.npy", candy)
np.save("results/revenue.npy", revenue)

fig, ax = plt.subplots(7)
fig.set_size_inches(5,7)

algs = ["Bare","DE","GA","GWO","Jaya","PSO","RO"]

for k,alg in enumerate(algs):
    ax[k].plot(milk[k], marker='o', color='k', linewidth=0.5)
    ax[k].plot(candy[k], marker='o', color='k', fillstyle='none', linewidth=0.5)
    ax[k].set(ylim=(0,25), ylabel=alg, xticks=[])

plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("results_ranking_plot.png", dpi=300)
plt.savefig("results_ranking_plot.eps", dpi=300)
plt.close()

print("Mean revenue by algorithm:")
best = 0
mx = -1
for k,alg in enumerate(algs):
    print("    %4s: $%0.2f (%5.2f)" % (alg, revenue[k].mean(), revenue[k].std(ddof=1)/np.sqrt(10)))
    if (revenue[k].mean() > mx):
        best = k
        mx = revenue[k].mean()
print()

print("t-test, best vs rest:")
for k,alg in enumerate(algs):
    if (k == best):
        continue
    _,p = ttest_ind(revenue[best],revenue[k])
    print("    %4s vs %4s: %0.5f" % (algs[best], algs[k], p))

