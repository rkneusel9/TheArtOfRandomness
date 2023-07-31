from RE import *
import os

rng = RE(mode="int", low=9999, high=99999999, seed=8675309)

os.system("mkdir songs")

for i in range(10):
    os.system("python3 melody_maker.py 36 songs/major%d 32 30000 bare major mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 songs/minor%d 32 30000 bare minor mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 songs/dorian%d 32 30000 bare dorian mt19937 %d" % (i,rng.random()))
for i in range(10):
    os.system("python3 melody_maker.py 36 songs/blues%d 32 30000 bare blues mt19937 %d" % (i,rng.random()))


