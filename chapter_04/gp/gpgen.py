#
#  file:  gpgen.py
#
#  Generate datasets for gp.py
#
#  RTK, 17-May-2022
#  Last update:  19-May-2022
#
################################################################

import matplotlib.pylab as plt
import numpy as np
import sys

if (len(sys.argv) == 1):
    print()
    print("gpgen <a> <b> <c> <d> <e> <f> <lo> <hi> <noise> <output>")
    print()
    print("    <a> .. <f> = ax^5 + bx^4 + cx^3 + dx^2 + ex + f, coefficients")
    print("    <lo>, <hi> = x range [lo,hi]")
    print("    <noise>    = multiplier on random noise (0=no noise)")
    print("    <output>   = output base name (.txt,.png), y then x")
    print()
    exit(0)

a,b,c,d,e,f = [float(i) for i in sys.argv[1:7]]
lo, hi = [float(i) for i in sys.argv[7:9]]
noise = float(sys.argv[9])
oname = sys.argv[10]

x = np.linspace(lo,hi,100)
y = a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f + noise*(np.random.random(100) - 0.5)
with open(oname+".txt", "w") as f:
    for i in range(len(x)):
        f.write("%0.7f  %0.7f\n" % (y[i],x[i]))

if (a != 0):
    n = 5
elif (b != 0):
    n = 4
elif (c != 0):
    n = 3
elif (d != 0):
    n = 2
elif (e != 0):
    n = 1
else:
    print("need at least x")
    exit(1)
p = np.polyfit(x,y,n)
yf = 0.0
s = "y = "
for i,t in enumerate(p):
    yf += t*x**(n-i)
    if ((n-i) == 0):
        if (t < 0):
            s += "$-%0.2f$" % (np.abs(t),)
        else:
            s += "$+%0.2f$" % (t,)
    elif ((n-i) == 1):
        if (t < 0):
            s += "$-%0.2fx$" % (np.abs(t),)
        else:
            s += "$+%0.2fx$" % (t,)
    else:
        if (t < 0):
            s += "$-%0.2fx^%d$" % (np.abs(t),n-i)
        else:
            s += "$+%0.2fx^%d$" % (t,n-i)

plt.plot(x,y, marker='o', linestyle='none')
plt.plot(x,yf)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(s)
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(oname+".png", dpi=300)
plt.show()

