import numpy as np
import matplotlib.pylab as plt

g = np.load("gbest.npy")
w = np.load("giter.npy")
n = 1000

x = np.arange(n)
y = np.zeros(n)

for i in range(len(g)):
    y[w[i]:] = g[i]

plt.plot(x,y, color='k')
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("gbest_plot.eps", dpi=300)
plt.show()

