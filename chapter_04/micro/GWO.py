#
#  file:  GWO.py
#
#  Grey wolf optimization
#
#  RTK, 23-Dec-2019
#  Last update:  24-May-2022
#
################################################################

import numpy as np
from RE import *

################################################################
#  GWO
#
class GWO:
    """Grey wolf optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 eta=2.0,         # scale factor for a
                 npart=10,        # number of particles in the swarm (> 3)
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
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
            "gbest": self.gbest,            # sequence of global best function values
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle id of global best
            "giter": self.giter,            # iteration number of global best
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
        self.vpos= np.zeros(self.npart)
        for i in range(self.npart):
            self.vpos[i] = self.obj.Evaluate(self.pos[i])

        #  Swarm bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []
        idx = np.argmin(self.vpos)
        self.gidx.append(idx)
        self.gbest.append(self.vpos[idx])
        self.gpos.append(self.pos[idx].copy())
        self.giter.append(0)

        #  1st, 2nd, and 3rd best positions
        idx = np.argsort(self.vpos)
        self.alpha = self.pos[idx[0]].copy()
        self.valpha= self.vpos[idx[0]]
        self.beta  = self.pos[idx[1]].copy()
        self.vbeta = self.vpos[idx[1]]
        self.delta = self.pos[idx[2]].copy()
        self.vdelta= self.vpos[idx[2]]


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
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        #  a from eta ... zero (default eta is 2)
        a = self.eta - self.eta*(self.iterations/self.max_iter)

        #  Update everyone
        for i in range(self.npart):
            A = 2*a*self.rng.random(self.ndim) - a
            C = 2*self.rng.random(self.ndim)
            Dalpha = np.abs(C*self.alpha - self.pos[i]) 
            X1 = self.alpha - A*Dalpha

            A = 2*a*self.rng.random(self.ndim) - a
            C = 2*self.rng.random(self.ndim)
            Dbeta = np.abs(C*self.beta - self.pos[i]) 
            X2 = self.beta - A*Dbeta

            A = 2*a*self.rng.random(self.ndim) - a
            C = 2*self.rng.random(self.ndim)
            Ddelta = np.abs(C*self.delta - self.pos[i]) 
            X3 = self.delta - A*Ddelta 
            
            self.pos[i,:] = (X1+X2+X3) / 3.0

        #  Keep in bounds
        if (self.bounds != None):
            self.pos = self.bounds.Limits(self.pos)

        #  Get objective function values and check for new leaders
        for i in range(self.npart):
            self.vpos[i] = self.obj.Evaluate(self.pos[i])
			
            #  new alpha?
            if (self.vpos[i] < self.valpha):
                self.vdelta = self.vbeta
                self.delta = self.beta.copy()
                self.vbeta = self.valpha
                self.beta = self.alpha.copy()
                self.valpha = self.vpos[i]
                self.alpha = self.pos[i].copy()

            #  new beta?
            if (self.vpos[i] > self.valpha) and (self.vpos[i] < self.vbeta):
                self.vdelta = self.vbeta
                self.delta = self.beta.copy()
                self.vbeta = self.vpos[i]
                self.beta = self.pos[i].copy()
            
            #  new delta?
            if (self.vpos[i] > self.valpha) and (self.vpos[i] < self.vbeta) and (self.vpos[i] < self.vdelta):
                self.vdelta = self.vpos[i]
                self.delta = self.pos[i].copy()

            #  is alpha new swarm best?
            if (self.valpha < self.gbest[-1]):
                self.gidx.append(i)
                self.gbest.append(self.valpha)
                self.gpos.append(self.alpha.copy())
                self.giter.append(self.iterations)

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


# end GWO.py

