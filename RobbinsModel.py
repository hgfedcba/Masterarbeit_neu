import copy

import numpy as np
from ModelDefinitions import sigma_dict, payoff_dict, mu_dict
import scipy.stats
import random
from AbstractMathematicalModel import AbstractMathematicalModel


class RobbinsModel(AbstractMathematicalModel):
    def __init__(self, N):
        self.__N = N  # X_0,..., X_(N-1)
        self.__reference_value = None
        self.t = self.get_time_partition()
        self.parameter_string = ""
        self.parameter_list = []
        self.__K = 0  # TODO: solve this better. This exists since K is the offset towards the origin for the nets   (f.e. K=0.5 :P)

    def update_parameter_string(self):
        parameter_string = "reference_value: ", round(self.__reference_value, 3), "N: ", self.__N

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)

        self.parameter_string = parameter_string + "\n"

    # antithetic is ignored
    def generate_paths(self, L, antithetic=None):
        """
        for l in range(L):
            out.append(np.random.uniform(low=0.0, high=1.0, size=None))
            # if l % 100000 == 0:
            #     print(l)
        """

        x = np.random.uniform(low=0.0, high=1.0, size=(L, self.__N)).tolist()

        y = []
        for l in range(L):
            y.append([])
            y[l].append([x[l][0]])
            for n in range(1, self.__N):
                y[l].append(copy.deepcopy(y[l][n - 1]))
                y[l][n].append(x[l][n])

        assert L == y.__len__()

        return y

    def get_time_partition(self, N=None, step_size=1):
        return range(1, self.__N+1)

    def getT(self):
        return self.__N

    def getN(self):
        return self.__N

    def getK(self):
        return self.__K

    def getprocess_dim(self):
        return 1

    def getpath_dim(self):
        return range(1, self.__N+1)

    def convert_NN_path_to_mathematical_path(self, x):
        # Dude, seriously
        """
        out = np.ones_like(x)
        for k in range(x.__len__()):
            out[k] *= x[k][k]
        """
        return np.array(x[-1])

    def getg(self, t, x):
        # (zeitstetige) stoppzeit t, pfad x
        assert t.sum() == 1

        h = self.convert_NN_path_to_mathematical_path(x)

        # Schritt 1: Ersetze in x jeden Wert mit dem entsprechenden Rang
        # TODO: (beachte das problem von 2 identischen werten)
        y = np.argsort(h)
        z1 = np.ones_like(y)
        z1[y] = np.arange(1, h.size + 1)
        """
        z = np.ones_like(y)
        for k in range(y.size):
            z[y[k]] = k+1
        """

        # Schritt 2: Bilde das Skalarprodukt von t und z1
        return np.dot(t, z1)

    def set_reference_value(self, v):
        self.__reference_value = v

    def get_reference_value(self):
        return self.__reference_value



