#
#  file:  RandomInitialize.py
#
#  Initialize a swarm uniformly within the bounds.
#
#  RTK, 07-Dec-2019
#  Last update:  14-May-2022
#
################################################################

import numpy as np
from RE import *

################################################################
#  RandomInitializer
#
class RandomInitializer:
    """Initialize a swarm uniformly"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, npart=10, ndim=3, bounds=None, rng=None):
        """Constructor"""

        self.npart = npart
        self.ndim = ndim
        self.bounds = bounds
        if (rng == None):
            self.rng = RE()
        else:
            self.rng = rng


    #-----------------------------------------------------------
    #  InitializeSwarm
    #
    def InitializeSwarm(self):
        """Return a randomly initialized swarm"""

        if (self.bounds == None):
            #  No bounds given, just use [0,1)
            self.swarm = self.rng.random((self.npart, self.ndim))
        else:
            #  Bounds given, use them
            self.swarm = np.zeros((self.npart, self.ndim))
            lo = self.bounds.Lower()
            hi = self.bounds.Upper()

            for i in range(self.npart):
                for j in range(self.ndim):
                    self.swarm[i,j] = lo[j] + (hi[j]-lo[j])*self.rng.random()        
            self.swarm = self.bounds.Limits(self.swarm)

        return self.swarm


# end RandomInitializer.py

