import numpy as np
import matplotlib.pylab as plt

d = np.loadtxt("data/noisy_exp.txt")
x = np.linspace(-3,3,300)
de = 0.35484**(x**2)
bare = 2.80857**(-x**2)
jaya = 7.95565**(-x**2)

plt.plot(d[:,1],d[:,0], linestyle='none', marker='o', color='k')
plt.plot(x,de, color='k', label='DE')
plt.plot(x,bare, color='k', linestyle='dashed', label='bare bones')
plt.plot(x,jaya, color='k', linestyle='dashdot', label='Jaya')

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="upper right")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("noisy_exp_fits_plot.eps", dpi=300)
plt.show()

