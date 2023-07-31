#
#  file:  melody_maker.py
#
#  Generate a melody
#
#  RTK, 29-Jun-2022
#  Last update:  30-Jun-2022
#
################################################################

import time
import os
import sys
import pickle

import numpy as np
from midiutil import MIDIFile
from PIL import Image

from Jaya import *
from GWO import *
from PSO import *
from DE import *
from RO import *
from GA import *
from RE import *

from RandomInitializer import *
from Bounds import *
from LinearInertia import *

# Note duration multiplier
M = 0.3

################################################################
#  MusicBounds
#
class MusicBounds(Bounds):
    """Subclass of Bounds to enforce note limits"""

    def __init__(self, lower, upper, rng):
        """Just call the superclass constructor"""
        super().__init__(lower, upper, enforce="resample", rng=rng)

    def Validate(self, p):
        """Enforce note and duration discretization"""

        i = 0
        while (i < p.shape[0]):
            note, duration = p[i:(i+2)]
            p[i] = int(note)
            p[i+1] = np.floor(duration)
            i += 2

        return p


################################################################
#  MusicObjective
#
class MusicObjective:
    """Measure the quality of the melody"""
    
    def __init__(self, note_lo, note_hi, mode):
        """Constructor"""

        self.mode = mode
        self.fcount = 0
        self.lo = note_lo
        self.hi = note_hi + 1

    def Durations(self, p):
        """Favor quarter and half notes"""

        d = p[1::2].astype("int32")
        dp = np.bincount(d, minlength=8)
        b = dp / dp.sum()
        a = np.array([0,0,100,0,60,0,20,0])
        a = a / a.sum()
        return np.sqrt(((a-b)**2).sum())

    def ModeNotes(self, notes, mode):
        """Return the distance between the nodes and the given mode"""

        modes = {
            "ionian":     [2,2,1,2,2,2,1],
            "dorian":     [2,1,2,2,2,1,2],
            "phrygian":   [1,2,2,2,1,2,2],
            "lydian":     [2,2,2,1,2,2,1],
            "mixolydian": [2,2,1,2,2,1,2],
            "aeolian":    [2,1,2,2,1,2,2],
            "locrian":    [1,2,2,1,2,2,2],
            "major":      [2,2,1,2,2,2,1],
            "minor":      [2,1,2,2,1,2,2],
            "penta":      [2,2,3,2,3,2,2],
            "blues":      [3,2,1,1,3,2,3],
        }
        
        m = modes[mode.lower()]

        #  Actual notes in this melody
        A = np.zeros(self.hi-self.lo+1)
        for i in range(notes.shape[0]):
            A[int(notes[i]-self.lo)] = 1

        #  Notes in the given mode based on this root
        B = np.zeros(self.hi-self.lo+1)
        note = int(notes[0])
        while (note <= self.hi):
            if (note <= self.hi):
                B[note-self.lo] = 1
            note += m[0]
            if (note <= self.hi):
                B[note-self.lo] = 1
            note += m[1]
            if (note <= self.hi):
                B[note-self.lo] = 1
            note += m[2]
            if (note <= self.hi):
                B[note-self.lo] = 1
            note += m[3]
            if (note <= self.hi):
                B[note-self.lo] = 1
            note += m[4]
            if (note <= self.hi):
                B[note-self.lo] = 1
            note += m[5]
            if (note <= self.hi):
                B[note-self.lo] = 1
            note += m[6]
            if (note <= self.hi):
                B[note-self.lo] = 1
        note = int(notes[0])
        while (note >= self.lo):
            if (note >= self.lo):
                B[note-self.lo] = 1
            note -= m[6]
            if (note >= self.lo):
                B[note-self.lo] = 1
            note -= m[5]
            if (note >= self.lo):
                B[note-self.lo] = 1
            note -= m[4]
            if (note >= self.lo):
                B[note-self.lo] = 1
            note -= m[3]
            if (note >= self.lo):
                B[note-self.lo] = 1
            note -= m[2]
            if (note >= self.lo):
                B[note-self.lo] = 1
            note -= m[1]
            if (note >= self.lo):
                B[note-self.lo] = 1
            note -= m[0]
            if (note >= self.lo):
                B[note-self.lo] = 1

        return A,B

    def Intervals(self, notes, mode):
        """Count valid third and fifth intervals"""

        #  Count major, minor thirds and fifths
        #  favoring thirds over fifths
        _,B = self.ModeNotes(notes, mode)
        minor = major = fifth = 0
        for i in range(len(notes)-1):
            x = int(notes[i]-self.lo)
            y = int(notes[i+1]-self.lo)
            if (B[x] == 1) and (B[y] == 1):
                if (abs(x-y) == 3):
                    minor += 1
                if (abs(x-y) == 4):
                    major += 1
                if (abs(x-y) == 7):
                    fifth += 1
        w = (3*minor + 3*major + fifth) / 7
        return 1.0 - w/len(notes)

    def Leaps(self, notes, mode):
        """Number of leaps"""

        _,B = self.ModeNotes(notes, mode)
        leaps = 0
        for i in range(len(notes)-1):
            x = int(notes[i]-self.lo)
            y = int(notes[i+1]-self.lo)
            if (B[x] == 1) and (B[y] == 1):
                if (abs(x-y) > 5):
                    leaps += 1
        return leaps / len(notes)

    def Distance(self, notes, mode):
        """Hamming distance between notes and mode notes"""

        A,B = self.ModeNotes(notes, mode)
        lo = int(notes.min() - self.lo)
        hi = int(notes.max() - self.lo)
        a = A[lo:(hi+2)]
        b = B[lo:(hi+2)]
        score = (np.logical_xor(a,b)*1).sum()
        score /= len(a)
        return score

    def Evaluate(self, p):
        """Evaluate a given melody"""

        self.fcount += 1
        s = self.Distance(p[::2], self.mode)    # distance from the mode
        d = self.Durations(p)                   # durations
        i = self.Intervals(p[::2], self.mode)   # intervals
        l = self.Leaps(p[::2], self.mode)       # leaps

        return 4*s+3*d+2*i+l


################################################################
#  StoreMelody
#
def StoreMelody(p, fname):
    """Write a melody to disk"""

    tempo = 120
    volume = 100
    m = MIDIFile(1)
    m.addTempo(0, 0, tempo)
    m.addProgramChange(0, 0, 0, 0)  # acoustic piano

    i = 0
    t = 0.0
    while (i < len(p)):
        note, duration = p[i:(i+2)]
        i += 2
        if (note == 57):
            m.addNote(0, 0, 21, t, M*duration, 0) # rest
        else:
            m.addNote(0, 0, int(note), t, M*duration, volume)
        t += M*duration

    with open(fname, "wb") as f:
        m.writeFile(f)


################################################################
#  DisplayMelody
#
def DisplayMelody(p):
    """Display a melody"""

    ans = ""
    i = 0
    while (i < len(p)):
        note, duration = p[i:(i+2)]
        i += 2
        ans += "%d,%0.2f " % (note, M*duration)
    ans += "\n"
    return ans


################################################################
#  PlayMelody
#
def PlayMelody(p):
    """Play a melody"""
    
    StoreMelody(p, "/tmp/xyzzy.mid")
    os.system("wildmidi /tmp/xyzzy.mid >/dev/null 2>/dev/null")


#
#  main
#
if (len(sys.argv) == 1):
    print()
    print("melody_maker <length> <outfile> <npart> <max_iter> <alg> <mode> [<kind> | <kind> <seed>]")
    print()
    print("  <length>   - number of notes in the melody")
    print("  <outdir>   - output directory")
    print("  <npart>    - swarm size")
    print("  <max_iter> - maximum number of iterations")
    print("  <alg>      - algorithm: PSO,DE,RO,GWO,JAYA,GA,BARE")
    print("  <mode>     - mode")
    print("  <kind>     - randomness source")
    print("  <seed>     - random seed")
    print()
    exit(0)

ndim = int(sys.argv[1])
ndim *= 2
outdir = sys.argv[2]
npart = int(sys.argv[3])
max_iter = int(sys.argv[4])
alg = sys.argv[5].upper()
mode = sys.argv[6].lower()

if (len(sys.argv) == 9):
    rng = RE(kind=sys.argv[7], seed=int(sys.argv[8]))
elif (len(sys.argv) == 8):
    rng = RE(kind=sys.argv[7])
else:
    rng = RE()

#  Ruthlessly overwrite any existing directory
os.system("rm -rf %s" % outdir)
os.system("mkdir %s" % outdir)

#  Create the bounds object
note_lo = 57  # 57 = rest (play w/zero volume)
note_hi = 81  # A5 
dur_lo  = 1   # multiples of M
dur_hi  = 5
lower = [note_lo, dur_lo] * ndim
upper = [note_hi, dur_hi] * ndim
b = MusicBounds(lower, upper, rng)

#  Create the objective function
music = MusicObjective(note_lo, note_hi, mode)

#  Build the swarm and optimize
ri = RandomInitializer(npart, ndim, bounds=b, rng=rng)

if (alg == "PSO"):
    swarm = PSO(obj=music, npart=npart, ndim=ndim, max_iter=max_iter, init=ri, bounds=b, rng=rng,
                inertia=LinearInertia())
elif (alg == "BARE"):
    swarm = PSO(obj=music, npart=npart, ndim=ndim, max_iter=max_iter, init=ri, bounds=b, bare=True, rng=rng)
elif (alg == "DE"):
    swarm = DE(obj=music, npart=npart, ndim=ndim, max_iter=max_iter, init=ri, bounds=b, rng=rng)
elif (alg == "RO"):
    swarm = RO(obj=music, npart=npart, ndim=ndim, max_iter=max_iter, init=ri, bounds=b, rng=rng)
elif (alg == "GWO"):
    swarm = GWO(obj=music, npart=npart, ndim=ndim, max_iter=max_iter, init=ri, bounds=b, rng=rng)
elif (alg == "JAYA"):
    swarm = Jaya(obj=music, npart=npart, ndim=ndim, max_iter=max_iter, init=ri, bounds=b, rng=rng)
elif (alg == "GA"):
    swarm = GA(obj=music, npart=npart, ndim=ndim, max_iter=max_iter, init=ri, bounds=b, rng=rng)
else:
    raise ValueError("Unknown swarm algorithm: %s" % alg)

st = time.time()
swarm.Optimize()
en = time.time()

results  = "\nMelody maker:\n\n"
results += "npart = %d\n" % npart
results += "niter = %d\n" % max_iter
results += "alg = %s\n" % alg
results += "Optimization time = %0.3f seconds\n" % (en-st,)

#  Store the final melody
res = swarm.Results()
melody = res["gpos"][-1]
results += DisplayMelody(melody) + "\n"
PlayMelody(melody)
StoreMelody(melody, outdir+("/melody_%s.mid" % alg))
np.save(outdir+("/melody_%s.npy" % alg), melody)
pickle.dump(res,open(outdir+("/melody_%s.pkl" % alg),"wb"))

results += "%d best updates, final objective value %0.4f\n\n" % (len(res["gbest"]), res["gbest"][-1])

mname = outdir+("/melody_%s.mid" % alg)
ec = os.system("musescore3 -o %s -T 0 %s >/dev/null 2>/dev/null" % (outdir+"/source.png", mname))
if (ec == 0):
    os.system("mv %s %s" % (outdir+"/source-1.png", outdir+"/score.png"))
try:
    im = np.array(Image.open(outdir+"/score.png"))
    im = Image.fromarray(255-im[:,:,3])
    im.save(outdir+"/score.png")
except:
    pass

print(results)
with open(outdir+"/README.txt","w") as f:
    f.write(results)

