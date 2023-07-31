#
#  file:  DE.py
#
#  Differential evolution
#
#  RTK, 09-Dec-2019
#  Last update:  14-May-2022
#
################################################################

import numpy as np
from RE import *

################################################################
#  DE
#
class DE:
    """Differential evolution"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 #  defaults from Tvrdik(2007) "Differential Evolution with Competitive
                 #  Setting of Control Parameters":
                 CR=0.5,          # cross-over probability
                 F=0.8,           # mutation factor, [0,2]
                 mode="rand",     # v1 variant: "rand" or "best"
                 cmode="bin",     # crossover variant: "bin" or "GA"
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None,     # swarm bounds object
                 rng = None):     # randomness source

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.bounds = bounds
        self.tol = tol
        self.CR = CR
        self.F = F
        self.mode = mode.lower()
        self.cmode = cmode.lower()
        self.tmode = False
        self.initialized = False
        if (rng == None):
            self.rng = RE()
        else:
            self.rng = rng


    #-----------------------------------------------------------
    #  Results
    #
    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,            # number of particles
            "ndim": self.ndim,              # number of dimensions 
            "max_iter": self.max_iter,      # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,                # tolerance value, if any
            "gbest": self.gbest,            # sequence of global best function values
            "giter": self.giter,            # iterations when global best updates happened
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle number for new global best
            "pos": self.pos,                # current particle positions
            "vpos": self.vpos,              # and objective function values
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos= self.Evaluate(self.pos)      # and objective function values

        #  Swarm bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []

        self.gidx.append(np.argmin(self.vpos))
        self.gbest.append(self.vpos[self.gidx[-1]])
        self.gpos.append(self.pos[self.gidx[-1]].copy())
        self.giter.append(0)


    #-----------------------------------------------------------
    #  Done
    #
    def Done(self):
        """Check if we are done"""

        if (self.done == None):
            if (self.tol == None):
                return (self.iterations == self.max_iter)
            else:
                return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)
        else:
            return self.done.Done(self.gbest,
                        gpos=self.gpos,
                        pos=self.pos,
                        max_iter=self.max_iter,
                        iteration=self.iterations)


    #-----------------------------------------------------------
    #  Evaluate
    #
    def Evaluate(self, pos):
        """Evaluate a set of positions"""

        p = np.zeros(self.npart)
        for i in range(self.npart):
            p[i] = self.obj.Evaluate(pos[i])
        return p


    #-----------------------------------------------------------
    #  Candidate
    #
    def Candidate(self, idx):
        """Return a candidate vector for the given index"""

        k = np.argsort(self.rng.random(self.npart))
        while (idx in k[:3]):
            k = np.argsort(self.rng.random(self.npart))
        
        v1 = self.pos[k[0]]
        v2 = self.pos[k[1]]
        v3 = self.pos[k[2]]

        if (self.mode == "best"):
            v1 = self.gpos[-1]
        elif (self.mode == "toggle"):
            if (self.tmode):
                self.tmode = False
                v1 = self.gpos[-1]
            else:
                self.tmode = True
        
        #  Donor vector
        v = v1 + self.F*(v2 - v3)

        #  Candidate vector
        u = np.zeros(self.ndim)
        I = int((self.ndim-1)*self.rng.random())

        if (self.cmode == "bin"):
            #  Bernoulli crossover
            for j in range(self.ndim):
                if (self.rng.random() <= self.CR) or (j == I):
                    u[j] = v[j]
                elif (j != I):
                    u[j] = self.pos[idx,j]
        else:
            #  GA-style crossover
            u = self.pos[idx].copy()
            u[I:] = v[I:]

        return u


    #-----------------------------------------------------------
    #  CandidatePositions
    #
    def CandidatePositions(self):
        """Return a set of candidate positions"""

        pos = np.zeros((self.npart, self.ndim))

        for i in range(self.npart):
            pos[i] = self.Candidate(i)

        if (self.bounds != None):
            pos = self.bounds.Limits(pos)

        return pos


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        new_pos = self.CandidatePositions() # get new candidate positions
        p = self.Evaluate(new_pos)          # and evaluate them

        #  For each particle
        for i in range(self.npart):
            if (p[i] < self.vpos[i]):               # is new position better?
                self.vpos[i] = p[i]                 # keep the function value
                self.pos[i] = new_pos[i]            # and new position
            if (p[i] < self.gbest[-1]):             # is new position global best?
                self.gbest.append(p[i])             # new position is new swarm best
                self.gpos.append(new_pos[i].copy()) # keep the position
                self.gidx.append(i)                 # particle number
                self.giter.append(self.iterations)  # and when it happened

        self.iterations += 1


    #-----------------------------------------------------------
    #  Optimize
    #
    def Optimize(self):
        """Run a full optimization and return the best"""

        self.Initialize()

        while (not self.Done()):
            self.Step()

        return self.gbest[-1], self.gpos[-1]


# end DE.py

