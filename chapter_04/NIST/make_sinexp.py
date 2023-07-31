#
#  file:  make_sinexp.py
#
#  Make the sinexp.txt dataset.
#
#  RTK, 31-Dec-2019
#  Last update:  31-Dec-2019
#
################################################################

import numpy as np

np.random.seed(8675309)

N = 40
x = 10.0*np.arange(N)/N
y = 2.0*np.sin(3.0*x) + 20.0*np.exp(-0.5*(x-8.0)**2/0.6) + 0.0005*(np.random.random()-0.5)

with open("sinexp.txt","w") as f:
    f.write("p[0]*np.sin(p[1]*x)+p[2]*np.exp(-0.5*(x-p[3])**2/p[4])\n")
    for i in range(N):
        f.write("%0.16f  %0.16f\n" % (y[i], x[i]))

