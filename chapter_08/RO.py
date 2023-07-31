#
#  file:  RO.py
#
#  Parallel random optimization
#
#  RTK, 07-Dec-2019
#  Last update:  15-May-2022
#
################################################################

import numpy as np
from RE import *

################################################################
#  normal -- Box-Muller generated normal samples
#
def normal(rng, mu=0, sigma=1):
    """Return N(m,s) samples"""

    if (normal.state):
        normal.state = False
        return sigma*normal.z2 + mu
    else:
        u1,u2 = rng.random(2)
        m = np.sqrt(-2.0*np.log(u1))
        z1 = m*np.cos(2*np.pi*u2)
        normal.z2 = m*np.sin(2*np.pi*u2)
        normal.state = True
        return sigma*z1 + mu

#  Set the initial state attribute
normal.state = False


################################################################
#  RO
#
class RO:
    """Parallel random optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 eta=0.1,         # max fractional change for candidate positions
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None,     # swarm bounds object
                 rng=None):       # randomness engine

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.bounds = bounds
        self.tol = tol
        self.eta = eta
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
            "eta": self.eta,                # max candidate fraction
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
        self.gpos.append(self.pos[self.gidx[-1]])
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
    #  CandidatePositions
    #
    def CandidatePositions(self):
        """Return a set of candidate positions"""
        
        n = []
        for i in range(self.npart * self.ndim):
            n.append(normal(self.rng)/5.0)
        n = np.array(n).reshape((self.npart, self.ndim))
        pos = self.pos + self.eta*self.pos*n

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
                self.gpos.append(new_pos[i])        # keep the position
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


# end RO.py

