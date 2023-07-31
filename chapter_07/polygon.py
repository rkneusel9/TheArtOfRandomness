#
#  file: polygon.py
#
#  Polygon fractals via the chaos game
#
#  RTK, 06-Jul-2022
#  Last update:  06-Jul-2022
#
################################################################

import sys
from RE import *
import turtle as tu
from matplotlib import cm

def Color(cmap,k,n):
    m = int(256*k/n)
    r,g,b,_ = [int(255*i) for i in cmap(m)]
    return "#%02X%02X%02X" % (r,g,b)

if (len(sys.argv) == 1):
    print()
    print("polygon <n>")
    print()
    print("  <n> - number of sides (n > 2)")
    print()
    exit(0)

n = max(int(sys.argv[1]),3)
w = 1.0
for k in range(1,1+int(n/4)):
    w += np.cos(2*np.pi*k/n)
r = 1 / (2*w)

angle = 2*np.pi / n
cmap = cm.get_cmap("hsv")

X = []
Y = []
for k in range(n):
    X.append(np.cos(k*angle))
    Y.append(np.sin(k*angle))

tu.mode('logo')
tu.speed(0)
tu.pu()
tu.getscreen().setup(600,600)
tu.getscreen().bgcolor('black')

x = X[0]
y = Y[0]
tu.color(Color(cmap,0,n))

if (len(sys.argv) == 3):
    rng = RE(mode='int', low=0, high=n, kind=sys.argv[2])
else:
    rng = RE(mode='int', low=0, high=n)

done = False

def Done():
    global done
    done = True

tu.onkeypress(Done)
tu.listen()

c = 0
while (not done):
    k = rng.random()
    x = r*(x + X[k])
    y = r*(y + Y[k])
    if (c > 2*n):
        tu.color(Color(cmap,k,n))
        tu.goto(80*n*x,80*n*y)
        tu.pd()
        tu.dot(1)
        tu.pu()
    c += 1

tu.ht()
tu.done()

