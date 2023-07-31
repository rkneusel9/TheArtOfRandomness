#
#  file:  RE.py
#
#  The randomness engine class
#
#  RTK, 22-Mar-2022
#  Last update:  28-Mar-2022
#
################################################################

import sys
import os
import numpy as np

try:
    import rdrand
    haveRDRAND = True
except:
    haveRDRAND = False


################################################################
#  RE
#
class RE:
    """Randomness Engine"""

    #-----------------------------------------------------------
    #  Fetch
    #
    def Fetch(self, N=1):
        """Fetch from a file wrapping as needed"""

        if (self.mode == "byte"):
            nbytes = N
        else:
            nbytes = 4*N  # float32 or int32

        b = []
        n = nbytes
        while (len(b) < nbytes):
            t = self.file.read(n)
            if (len(t) < n):
                n = n - len(t)
                self.file.close()
                self.file = open(self.kind, "rb")
            b += t

        if (self.mode == "byte"):
            v = np.array(b, dtype="uint8")
        else:
            v = np.frombuffer(bytearray(b), dtype="uint32")
            v = v / (1 << 32)
            if (self.mode == "float"):
                v = (self.high - self.low)*v + self.low
            elif (self.mode == "int"):
                v = ((self.high - self.low)*v).astype("int64") + self.low
            elif (self.mode == "byte"):
                v = np.floor(256*v + 0.5).astype("uint8")
            else:
                v = np.floor(v + 0.5).astype("uint8")
        
        return v


    #-----------------------------------------------------------
    #  Park and Miller MINSTD
    #
    def MINSTD(self, N):
        """Return a [0,1) vector of N elements"""

        v = np.zeros(N)
        for i in range(N):
            self.seed = (48271 * self.seed) % 2147483647
            v[i] = self.seed * 4.656612875245797e-10
        return v


    #-----------------------------------------------------------
    #  Urandom
    #
    def Urandom(self, N):
        """Read values from urandom"""

        #  fetch bytes
        with open("/dev/urandom", "rb") as f:
            b = bytearray(f.read(4*N))
         
        #  convert to [0,1) floats. N.B. if bytes wanted, read 
        #  urandom directly, this step is slow and returns float32
        #  values
        return np.frombuffer(b, dtype="uint32") / (1<<32)


    #-----------------------------------------------------------
    #  RDRAND
    #
    def RDRAND(self, N):
        """Use rdrand module, or fall back to default_rng"""

        if (not haveRDRAND):
            print("Warning: rdrand module not found, using NumPy instead", file=sys.stderr)
            return np.random.Generator(np.random.PCG64()).random(N)

        #  Get floats.  If bytes desired, you're better off using
        #  rdrand.rdrand_get_bytes() directly.  Faster and better stats.
        v = np.zeros(N)
        rng = rdrand.RdRandom()
        for i in range(N):
            v[i] = rng.random()
        return v


    #-----------------------------------------------------------
    #  Quasirandom
    #
    def Quasirandom(self, N):
        """Return a [0,1) vector of quasirandom values"""

        def Halton(i,b):
            """Return i-th Halton number for the given base"""
            f = 1.0
            r = 0
            while (i > 0):
                f = f/b
                r = r + f*(i % b)
                i = np.floor(i/float(b))
            return r

        v = []
        while (len(v) < N):
            v.append(Halton(self.qnum, self.base))
            self.qnum += 1
        return np.array(v)


    #-----------------------------------------------------------
    #  NumPyGen
    #
    def NumPyGen(self, N):
        """Use a NumPy-supported generator"""

        return self.g.random(N)


    #-----------------------------------------------------------
    #  random
    #
    def random(self, N=1):
        """Return a vector of N values"""
    
        if (not self.disk):
            #  Get a [0,1) vector
            v = self.generators[self.kind](N)

            #  Process
            if (self.mode == "float"):
                v = (self.high - self.low)*v + self.low
            elif (self.mode == "int"):
                v = ((self.high - self.low)*v).astype("int64") + self.low
            elif (self.mode == "byte"):
                v = np.floor(256*v + 0.5).astype("uint8")
            else:
                v = np.floor(v + 0.5).astype("uint8")
        else:
            #  From a disk file
            v = self.Fetch(N)

        return v[0] if (N == 1) else v 


    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, mode="float", kind="pcg64", seed=None, low=0, high=1, base=2):
        """Constructor"""

        #  Generators
        self.generators = {
            "pcg64"  : self.NumPyGen,
            "mt19937": self.NumPyGen,
            "minstd" : self.MINSTD, 
            "quasi"  : self.Quasirandom,
            "urandom": self.Urandom,
            "rdrand" : self.RDRAND,
        }

        #  Keep arguments
        self.mode = mode            # output type: "float", "int", "byte", "bit"
        self.kind = kind            # generator type: "pcg64", "mt19937", "minstd", "quasi", "urandom", "rdrand" | filename
        self.seed = seed            # integer seed or nothing
        self.low  = low             # minimum output value (inclusive)
        self.high = high            # maximum output value (exclusive)
        self.base = base            # quasirandom base (must be a prime)
        self.disk = False           # True if reading a file

        #  Configure generator
        if (self.kind == "pcg64"):
            self.g = np.random.Generator(np.random.PCG64(seed))
        elif (self.kind == "mt19937"):
            self.g = np.random.Generator(np.random.MT19937(seed))
        elif (self.kind == "minstd"):
            if (seed == None):
                self.seed = np.random.randint(1,93123544)
        elif (self.kind == "quasi"):
            if (seed == None):
                self.qnum = 0  # for quasirandom numbers in order
            elif (seed < 0):
                self.qnum = np.random.randint(0,10000)
            else:
                self.qnum = seed
        elif (self.kind == "urandom") or (self.kind == "rdrand"):
            pass
        else:
            self.disk = True
            self.file = open(self.kind, "rb")


