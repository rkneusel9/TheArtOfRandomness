#
#  file:  Bounds.py
#
#  Base bounds class.
#
#  RTK, 07-Dec-2019
#  Last update:  14-May-2022
#
################################################################

import numpy as np
from RE import *

################################################################
#  Bounds
#
class Bounds:
    """Base bounds class"""

    #-----------------------------------------------------------
    #  __init__
    #
    #  Supply lower and upper bounds for each dimension.
    #
    def __init__(self, lower, upper, enforce="clip", rng=None):
        """Constructor"""

        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.enforce = enforce.lower() # clip | resample
        if (rng == None):
            self.rng = RE()
        else:
            self.rng = rng


    #-----------------------------------------------------------
    #  Upper
    #
    #  Return a vector of the per dimension upper limits.
    #
    def Upper(self):
        """Upper bounds"""

        return self.upper


    #-----------------------------------------------------------
    #  Lower
    #
    def Lower(self):
        """Lower bounds"""

        return self.lower


    #-----------------------------------------------------------
    #  Limits
    #
    def Limits(self, pos):
        """Apply the selected boundary conditions"""

        npart, ndim = pos.shape

        for i in range(npart):
            if (self.enforce == "resample"):
                for j in range(ndim):
                    if (pos[i,j] <= self.lower[j]) or (pos[i,j] >= self.upper[j]):
                        pos[i,j] = self.lower[j] + (self.upper[j]-self.lower[j])*self.rng.random()
            else:  # clip
                for j in range(ndim):
                    if (pos[i,j] <= self.lower[j]):
                        pos[i,j] = self.lower[j]
                    if (pos[i,j] >= self.upper[j]):
                        pos[i,j] = self.upper[j]
            
            #  Also validate
            pos[i] = self.Validate(pos[i])

        return pos


    #-----------------------------------------------------------
    #  Validate
    #
    #  For example, override this to enforce a discrete position
    #  for a particular vector.
    #
    def Validate(self, pos):
        """Validate a given position vector"""

        return pos


#  end Bounds.py

