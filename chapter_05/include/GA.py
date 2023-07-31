#
#  file:  GA.py
#
#  A genetic algorithm
#
#  RTK, 29-Dec-2019
#  Last update:  14-May-2022
#
################################################################

import numpy as np
from RE import *

################################################################
#  GA
#
class GA:
    """Genetic algorithm"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 CR=0.8,          # cross-over probability
                 F=0.05,          # mutation probability
                 top=0.5,         # top fraction (only breed with the top fraction)
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None,     # swarm bounds object
                 rng=None):       # randomness source

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
        self.top = top
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
    #  Mutate
    #
    def Mutate(self, idx):
        """Return a mutated position vector"""

        j = int(self.ndim*self.rng.random())
        if (self.bounds != None):
            self.pos[idx,j] = self.bounds.lower[j] + self.rng.random()*(self.bounds.upper[j]-self.bounds.lower[j])
        else:
            lower = self.pos[:,j].min()
            upper = self.pos[:,j].max()
            self.pos[idx,j] = lower + self.rng.random()*(upper-lower)


    #-----------------------------------------------------------
    #  Crossover
    #
    def Crossover(self, a, idx):
        """Mate with another swarm member"""

        #  Get the partner in the top set
        n = int(self.top*self.npart)
        b = idx[int(n*self.rng.random())]
        while (a==b):
            b = idx[int(n*self.rng.random())]

        #  Random cut-off position
        d = int(self.ndim*self.rng.random())

        #  Crossover
        t = self.pos[a].copy()
        t[d:] = self.pos[b,d:]
        self.pos[a] = t.copy()


    #-----------------------------------------------------------
    #  Evolve
    #
    def Evolve(self):
        """Evolve the swarm"""

        idx = np.argsort(self.vpos)

        for k,i in enumerate(idx):
            if (k == 0):
                continue    #  leave the best one alone
            if (self.rng.random() < self.CR):
                #  Breed this one with one of the better particles
                self.Crossover(i, idx)
            if (self.rng.random() < self.F):
                #  Random mutation
                self.Mutate(i)

        if (self.bounds != None):
            self.pos = self.bounds.Limits(self.pos)


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        self.Evolve()                               # evolve the swarm
        self.vpos = self.Evaluate(self.pos)         # and evaluate the new positions

        #  For each particle
        for i in range(self.npart):
            if (self.vpos[i] < self.gbest[-1]):         # is new position global best?
                self.gbest.append(self.vpos[i])         # new position is new swarm best
                self.gpos.append(self.pos[i].copy())    # keep the position
                self.gidx.append(i)                     # particle number
                self.giter.append(self.iterations)      # and when it happened

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


# end GA.py

