#
#  file:  bad_sample_test.py
#
#  Generate a plot of sample distance from the population means as
#  a function of the sample size
#
#  RTK, 02-Aug-2022
#  Last update:  02-Aug-2022
#
################################################################

import os
import numpy as np
import matplotlib.pylab as plt

res = []
for s in [10,20,30,40,50,100,200,300,400,500,1000,2000,3000,4000,5000]:
    cmd = "python3 bad_sample.py 10000 %d 40 mt19937 10021 >/tmp/bad" % s
    os.system(cmd)
    x,y,z = [float(i) for i in open("/tmp/bad").read()[:-1].split()]
    res.append([x,y,z])
res = np.array(res)

plt.errorbar(res[:,0],res[:,1],res[:,2], color='k', marker='o', linewidth=0.6, capsize=2, fillstyle='none')
plt.xlabel("Sample size")
plt.ylabel("Distance from population mean")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("bad_sample_plot.png", dpi=300)
plt.savefig("bad_sample_plot.eps", dpi=300)
plt.show()

