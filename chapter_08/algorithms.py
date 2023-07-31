from RE import *
import os

rng = RE(mode="int", low=9999, high=99999999, seed=8675309)

os.system("mkdir algorithms")

for i in range(10):
    os.system("python3 melody_maker.py 36 algorithms/bare%d 20 10000 bare lydian mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 algorithms/de%d 20 10000 de lydian mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 algorithms/gwo%d 20 10000 gwo lydian mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 algorithms/ga%d 20 10000 ga lydian mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 algorithms/ro%d 20 10000 ro lydian mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 algorithms/pso%d 20 10000 pso lydian mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 algorithms/jaya%d 20 10000 jaya lydian mt19937 %d" % (i,rng.random()))

