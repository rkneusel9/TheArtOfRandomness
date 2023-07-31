#
#  file: elm_test_results.py
#
#  Evaluate test results
#
#  RTK, 23-Jun-2022
#  Last update, 23-Jun-2022
#
####################################################

import numpy as np
import matplotlib.pylab as plt

acts = ["relu", "sigmoid", "tanh", "cube", "absolute", "recip", "identity"]
nodes = [10,20,30,40,50,100,150,200,250,300,350,400]

res = np.load("elm_test_results.npy")

markers = ["o","s","^",">","<","d","X"]

for i,act in enumerate(acts):
    x = []
    y = []
    e = []
    for j,n in enumerate(nodes):
        x.append(n)
        y.append(res[i,j,:].mean())
        e.append(res[i,j,:].std(ddof=1)/np.sqrt(res.shape[2]))
    plt.errorbar(x,y,e, marker=markers[i], linewidth=0.5, color='k', fillstyle='none', label=act)

plt.xlim((0,525))
plt.xlabel("nodes")
plt.ylabel("accuracy")
plt.legend(loc="upper right")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("elm_test_results_plot.png", dpi=300)
plt.savefig("elm_test_results_plot.eps", dpi=300)
plt.show()

