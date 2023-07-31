#
#  file: sierpinski.py
#
#  Algorithmic Sierpinski triangle
#
#  RTK, 28-Jun-2022
#  Last update:  28-Jun-2022
#
################################################################

import sys
from RE import *
import turtle as tu

tu.mode('logo')
tu.speed(0)
tu.pu()
tu.getscreen().setup(500,500)
tu.getscreen().bgcolor('black')

X = [-200,0,200]
Y = [-200,200,-200]
x = X[0]
y = Y[0]
colors = ['#E7FFAC','#ACE7FF','#97A2FF']
tu.color(colors[0])

if (len(sys.argv) == 2):
    rng = RE(mode='int', low=0, high=3, kind=sys.argv[1])
else:
    rng = RE(mode='int', low=0, high=3)

done = False

def Done():
    global done
    done = True

tu.onkeypress(Done)
tu.listen()

while (not done):
    n = rng.random()
    x = 0.5*(x + X[n])
    y = 0.5*(y + Y[n])
    tu.color(colors[n])
    tu.goto(x,y)
    tu.pd()
    tu.dot(1)
    tu.pu()

tu.ht()
tu.done()

