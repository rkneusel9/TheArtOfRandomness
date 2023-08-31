#
#  file:  curfit_example.py
#
#  A basic example of curve fitting.
#
#  RTK, 14-May-2022
#  Last update:  14-May-2022
#
################################################################

import numpy as np
import matplotlib.pylab as plt
from RE import *

#  Generate some random data that, roughly, follows a quadratic
x = np.linspace(0,10,15)
y = -3*x**2 + 12*x + 4 + 18*(RE(seed=73939133).random(len(x))-0.5)

#  Fit a quadratic: a*x**2 + b*x + c using NumPy
p = np.polyfit(x,y,2)
yf = p[0]*x**2 + p[1]*x + p[2]

#  Plot
plt.plot(x,y, marker='+', color='k', linestyle='none')
plt.plot(x,yf, color='k', label="$%0.3f x^2 + %0.3f x + %0.3f$" % (p[0],p[1],p[2]))
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc="lower left")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("curfit_example_plot.png", dpi=300)
plt.savefig("curfit_example_plot.eps", dpi=300)
plt.show()

#for i in range(len(x)):
#    print("%0.7f %0.7f" % (y[i],x[i]))

