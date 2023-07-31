#
#  file:  LinearInertia.py
#
#  Linear inertia class for canonical PSO.
#  Python 3.X
#
#  RTK, 09-Dec-2019
#  Last update:  09-Dec-2019
#
################################################################

################################################################
#  LinearInertia
#
class LinearInertia:
    """A linear inertia class"""

    #-----------------------------------------------------------
    #  __init___
    #
    def __init__(self, hi=0.9, lo=0.6):
        """Constructor"""
        
        if (hi > lo):
            self.hi = hi
            self.lo = lo
        else:
            self.hi = lo
            self.lo = hi


    #-----------------------------------------------------------
    #  CalculateW
    #
    def CalculateW(self, w0, iterations, max_iter):
        """Return a weight value"""

        return self.hi - (iterations/max_iter)*(self.hi-self.lo)


#  end LinearInertia.py

