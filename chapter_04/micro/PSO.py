#
#  file:  PSO.py
#
#  Particle swarm optimization.  Canonical and bare-bones w/
#  global neighborhood.
#
#  RTK, 08-Dec-2019
#  Last update:  14-May-2022
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
#  PSO
#
class PSO:
    """Particle swarm optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 c1=1.49,         # cognitive parameter
                 c2=1.49,         # social parameter
                 #  best if w > 0.5*(c1+c2) - 1:
                 w=0.729,         # base velocity decay parameter
                 inertia=None,   # velocity weight decay object (None == constant)
                 #  Bare-bones from:
                 #    Kennedy, James. "Bare bones particle swarms." In Proceedings of 
                 #    the 2003 IEEE Swarm Intelligence Symposium. SIS'03 (Cat. No. 03EX706), 
                 #    pp. 80-87. IEEE, 2003.
                 bare=False,      # if True, use bare-bones update
                 bare_prob=0.5,   # probability of updating a particle's component
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 ring=False,      # use ring topology if True
                 neighbors=2,     # number of particle neighbors for ring, must be even
                 vbounds=None,    # velocity bounds object
                 bounds=None,     # swarm bounds object
                 rng=None):       # randomness source

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.vbounds = vbounds
        self.bounds = bounds
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.bare = bare
        self.bare_prob = bare_prob
        self.inertia = inertia
        self.ring = ring
        self.neighbors = neighbors
        self.initialized = False
        if (rng == None):
            self.rng = RE()
        else:
            self.rng = rng

        if (ring) and (neighbors > npart):
            self.neighbors = npart


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
            "c1": self.c1,                  # cognitive parameter
            "c2": self.c2,                  # social parameter
            "w": self.w,                    # base velocity decay parameter
            "tol": self.tol,                # tolerance value, if any
            "gbest": self.gbest,            # sequence of global best function values
            "giter": self.giter,            # iterations when global best updates happened
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle number for new global best
            "pos": self.pos,                # current particle positions
            "vel": self.vel,                # velocities
            "xpos": self.xpos,              # per particle best positions
            "xbest": self.xbest,            # per particle bests
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()       # initial swarm positions
        self.vel = np.zeros((self.npart, self.ndim)) # initial velocities
        self.xpos = self.pos.copy()                  # these are the particle bests
        self.xbest= self.Evaluate(self.pos)          # and objective function values

        #  Swarm and particle bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []

        self.gidx.append(np.argmin(self.xbest))
        self.gbest.append(self.xbest[self.gidx[-1]])
        self.gpos.append(self.xpos[self.gidx[-1]].copy())
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
    #  RingNeighborhood
    #
    def RingNeighborhood(self, n):
        """Return a list of particles in the neighborhood of n"""

        idx = np.array(range(n-self.neighbors//2,n+self.neighbors//2+1))
        i = np.where(idx >= self.npart)
        if (len(i) != 0):
            idx[i] = idx[i] % self.npart
        i = np.where(idx < 0)
        if (len(i) != 0):
            idx[i] = self.npart + idx[i]

        return idx


    #-----------------------------------------------------------
    #  NeighborhoodBest
    #
    def NeighborhoodBest(self, n):
        """Return neighborhood best for particle n"""

        if (not self.ring):
            return self.gbest[-1], self.gpos[-1]

        # Using a ring, return best known position of the neighborhood
        lbest = 1e9
        for i in self.RingNeighborhood(n):
            if (self.xbest[i] < lbest):
                lbest = self.xbest[i]
                lpos = self.xpos[i]

        return lbest, lpos


    #-----------------------------------------------------------
    #  BareBonesUpdate
    #
    def BareBonesUpdate(self):
        """Apply a bare-bones update to the positions"""

        pos = np.zeros((self.npart, self.ndim))

        for i in range(self.npart):
            lbest, lpos = self.NeighborhoodBest(i)
            for j in range(self.ndim):
                if (self.rng.random() < self.bare_prob):
                    m = 0.5*(lpos[j] + self.xpos[i,j])
                    s = np.abs(lpos[j] - self.xpos[i,j])
                    pos[i,j] = normal(self.rng, m,s)
                else:
                    pos[i,j] = self.xpos[i,j]

        return pos


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        #  Weight for this iteration
        if (self.inertia != None):
            w = self.inertia.CalculateW(self.w, self.iterations, self.max_iter)
        else:
            w = self.w

        if (self.bare):
            #  Bare-bones position update
            self.pos = self.BareBonesUpdate()
        else:
            #  Canonical position/velocity update
            for i in range(self.npart):
                lbest, lpos = self.NeighborhoodBest(i)
                c1 = self.c1 * self.rng.random(self.ndim)
                c2 = self.c2 * self.rng.random(self.ndim)
                self.vel[i] = w*self.vel[i] +                    \
                              c1*(self.xpos[i] - self.pos[i]) +  \
                              c2*(lpos - self.pos[i])

            #  Keep velocities bounded
            if (self.vbounds != None):
                self.vel = self.vbounds.Limits(self.vel)

            #  Update the positions
            self.pos = self.pos + self.vel

        #  Keep positions bounded
        if (self.bounds != None):
            self.pos = self.bounds.Limits(self.pos)

        #  Evaluate the new positions
        p = self.Evaluate(self.pos)

        #  Check if any new particle and swarm bests
        for i in range(self.npart):
            if (p[i] < self.xbest[i]):                  # is new position a particle best?
                self.xbest[i] = p[i]                    # keep the function value
                self.xpos[i] = self.pos[i]              # and position
            if (p[i] < self.gbest[-1]):                 # is new position global best?
                self.gbest.append(p[i])                 # new position is new swarm best
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


# end PSO.py

