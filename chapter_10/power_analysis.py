#
#  file:  power_analysis.py
#
#  Use statsmodels to do a t-test power analysis to
#  determine the number of subjects needed to achieve
#  a desired effect.
#
#  RTK, 13-Aug-2022
#  Last updated:  13-Aug-2022
#
################################################################

import numpy as np
from statsmodels.stats.power import TTestIndPower

alpha = 0.05
power = 0.9
effect = ((np.arange(9)+1)/10)[::-1]

for i in range(9):
    nsubj = TTestIndPower().solve_power(effect[i], power=power, alpha=alpha)
    print("%0.1f: %4d subjects" % (effect[i], nsubj))

