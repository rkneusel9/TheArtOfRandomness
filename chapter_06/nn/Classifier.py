#
#  file: nn.py
#
#  Neural network initialization experiments
#
#  RTK, 09-Jun-2022
#  Last update:  09-Jun-2022
#
################################################################

import numpy as np
from sklearn.neural_network import MLPClassifier
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
#  Classifier
#
class Classifier(MLPClassifier):
    def _init_coef(self, fan_in, fan_out, dtype):
        """Override sklearn's version"""

        def normvec(fan_in, fan_out):
            vec = np.zeros(fan_in*fan_out)
            for i in range(fan_in*fan_out):
                vec[i] = normal(self.rng)
            return vec.reshape((fan_in,fan_out))

        if (self.init_scheme == 0):
            return super(Classifier, self)._init_coef(fan_in, fan_out, dtype)
        elif (self.init_scheme == 1):
            vec = self.rng.random(fan_in*fan_out).reshape((fan_in,fan_out))
            weights = 0.01*(vec-0.5)
            biases = np.zeros(fan_out)
        elif (self.init_scheme == 2):
            weights = 0.005*normvec(fan_in, fan_out)
            biases = np.zeros(fan_out)
        elif (self.init_scheme == 3):
            weights = normvec(fan_in, fan_out)*np.sqrt(2.0/fan_in)
            biases = np.zeros(fan_out)

        return weights.astype(dtype, copy=False), biases.astype(dtype, copy=False)

#
#  add attributes: init_scheme and rng to Classifier object before using
#


