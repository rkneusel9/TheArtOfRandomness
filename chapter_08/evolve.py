from RE import *
import os

os.system("rm -rf xyzzy; mkdir evolve_results")
cmd = "python3 melody_maker.py 20 xyzzy 20 %d bare major pcg64 73939133"

for limit in [1,10,100,1000,5000,10000,50000]:
    os.system(cmd % limit)
    os.system("mv xyzzy/melody_BARE.mid evolve_results/melody%06d.mid" % limit)
    os.system("mv xyzzy/score.png evolve_results/score%06d.png" % limit)

os.system("rm -rf xyzzy")

#       1   4.4915
#      10   3.4341
#     100   2.6332
#    1000   1.8832
#    5000   1.7923
#   10000   1.7494
#   50000   1.6637

