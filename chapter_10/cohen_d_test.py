#
#  file: cohen_d_test.py
#
#  Find approximate study sizes to show the desired treatment effect
#  at a level where the highest 10% group has a mean p-value < 0.05.
#
#  RTK, 13-Aug-2022
#  Last update:  13-Aug-2022
#
################################################################
import os

def RunTest(beta, nsubj):
    cmd = "python3 design.py 100000 %d %0.1f 20 2 0 minstd 6809 >/tmp/xyzzy"
    os.system(cmd % (nsubj,beta))
    lines = [i[:-1] for i in open("/tmp/xyzzy")]
    pv = float(lines[2].split()[-1])
    d =  float(lines[5].split()[-1])
    return pv,d

base = 10
for beta in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
    pvalue = 10.0
    k = 1

    while (pvalue > 0.05):
        pvalue,d = RunTest(beta, k*base)
        k += 1

    print("%0.1f: %3d subjects, p=%0.8f, d=%0.4f" % (beta, k*base, pvalue, d), flush=True)

