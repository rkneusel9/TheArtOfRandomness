#
#  file:  MiCRO.py
#
#  Minimally Conscious Random Optimization
#
#  RTK, 17-May-2022
#  Last update:  20-May-2022
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
#  MiCRO
#
class MiCRO:
    """minimally conscious random optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 eta=0.1,         # max fractional change for candidate positions
                 glimpse=0.01,    # probability a particle will look up
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
        self.glimpse = glimpse
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
        """Evaluate the positions"""

        v = []
        for i in range(self.npart):
            v.append(self.obj.Evaluate(pos[i]))
        return np.array(v)


    #-----------------------------------------------------------
    #  UpdatePositions
    #
    def UpdatePositions(self):
        """Return a set of new positions"""

        for i in range(self.npart):
            graze = True
            if (self.rng.random() < self.glimpse):
                #  Look up
                idx = np.where(self.vpos < self.vpos[i])[0]
                if (len(idx) != 0):
                    graze = False
                    k = int(len(idx)*self.rng.random())
                    n = []
                    for j in range(self.ndim):
                        n.append(normal(self.rng)/5.0)
                    self.pos[i] = self.pos[k] + 2*self.eta*self.pos[k]*np.array(n)
                    self.vpos[i] = self.obj.Evaluate(self.pos[i])

            if (graze):
                #  Continue grazing
                n = []
                for j in range(self.ndim):
                    n.append(normal(self.rng)/5.0)
                pos = self.pos[i] + self.eta*self.pos[i]*np.array(n)
                vpos = self.obj.Evaluate(pos)
                if (vpos < self.vpos[i]):
                    self.pos[i] = pos
                    self.vpos[i] = vpos

        #  Enforce limits and get final objective function values
        if (self.bounds != None):
            self.pos = self.bounds.Limits(self.pos)

        self.vpos = self.Evaluate(self.pos)


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        self.UpdatePositions()  #  update self.pos and self.vpos

        #  Check for new global bests
        for i in range(self.npart):
            if (self.vpos[i] < self.gbest[-1]):     # is new position global best?
                self.gbest.append(self.vpos[i])     # new position is new swarm best
                self.gpos.append(self.pos[i])       # keep the position
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


# end MiCRO.py

