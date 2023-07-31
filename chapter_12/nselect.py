#  Select a few samples from the standard normal, N(0,1),
#  and plot them

import numpy as np
import matplotlib.pylab as plt

#  select 11 values
np.random.seed(6502)
z = np.random.normal(0, 1, size=30)

#  plot them as points on the x-axis
x = np.linspace(-3.5,3.5,10000)
y = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

plt.plot(x,y, color='k')
for v in z:
    plt.plot([v,v],[0,0], marker='o', linestyle='none', markersize=4, color='k')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.savefig('nselect.png', dpi=300)
plt.savefig('nselect.eps', dpi=300)
plt.show()

