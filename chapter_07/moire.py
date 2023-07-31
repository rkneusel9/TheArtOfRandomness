#
#  file: moire.py
#
#  Recreate Brian's Theme
#
#  RTK, 28-Jun-2022
#  Last update:  28-Jun-2022
#
##################################################################

import time
import turtle as tu
from RE import *

def Line(x0,y0,x1,y1,color):
    tu.color('white')
    tu.pu()
    tu.goto(x0,y0)
    tu.pd()
    tu.goto(x1,y1)
    tu.color(color)
    tu.goto(x0,y0)
    tu.pu()


tu.speed(0)
tu.ht()
tu.getscreen().setup(500,500)
tu.getscreen().bgcolor('black')

x = np.linspace(-200,200,400)
y = np.linspace(-200,200,400)

while (True):
    tu.clear()

    X,Y = RE(mode='int', low=-100, high=100).random(2)
    step = RE(mode='int', low=2, high=9).random()
    r,g,b = RE(mode='int', low=1, high=256).random(3)
    color = "#%02x%02x%02x" % (r,g,b)
    tu.color(color)

    for i in range(0,400,step):
        Line(X,Y, x[i],y[0], color)
        Line(X,Y, x[0],y[i], color)
        Line(X,Y, x[i],y[-1], color)
        Line(X,Y, x[-1],y[i], color)

    time.sleep(4)

tu.done()

