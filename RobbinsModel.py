import copy

import numpy as np
import pytest
import torch
import Util

from ModelDefinitions import sigma_dict, payoff_dict, mu_dict
import scipy.stats
import random
from AbstractMathematicalModel import AbstractMathematicalModel


# TODO: important note: Ich maximiere den Rang statt ihn zu minimieren. Es gibt N+1 Ränge
class RobbinsModel(AbstractMathematicalModel):
    def __init__(self, N):
        self.__N = N  # X_0,..., X_(N)
        self.__reference_value_upper = None
        self.__reference_value_lower = None
        self.t = self.get_time_partition()
        self.parameter_string = ""
        self.parameter_list = []
        self.__K = 0  # solve this better. This exists since K is the offset towards the origin for the nets   (f.e. K=0.5 :P)

    def update_parameter_string(self, main_pc):
        parameter_string = "Robbins Model with unterem Referenzwert: ", round(self.__reference_value_upper, 3), "oberen Referenzwert: ", round(self.__reference_value_lower, 3), "N: ", self.__N+1,\
                           "auf dem " + main_pc

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)

        self.parameter_string = parameter_string + "\n"

    # antithetic is ignored
    def generate_paths(self, L, antithetic=None, sorted=False):
        """
        for l in range(L):
            out.append(np.random.uniform(low=0.0, high=1.0, size=None))
            # if l % 100000 == 0:
            #     print(l)
        """

        x = np.random.uniform(low=0.0, high=1.0, size=(L, self.__N+1)).tolist()

        y = []
        for l in range(L):
            y.append([])
            y[l].append([x[l][0]])
            for n in range(1, self.__N+1):
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

    def convert_multiple_NN_paths_to_mathematical_paths(self, x):
        l = len(x[0])
        return [np.array(y[-1]).reshape((1, l)) for y in x]

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

    def calculate_payoffs(self, U, x, g, t):
        if isinstance(U, np.ndarray):
            return self.getg(U, x)
        assert torch.sum(torch.tensor(U)).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(torch.tensor(U)).item())

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

        # Dieser Schritt ist für Alg10, damit nur so viele Werte betrachtet werden wie U enthält
        z2 = z1[-len(U):]

        h = torch.matmul(U, torch.tensor(z2, dtype=torch.float))

        # Schritt 2: Bilde das Skalarprodukt von t und z1
        return torch.matmul(U, torch.tensor(z2, dtype=torch.float))

    def set_reference_value(self):
        if self.getN()>12:
            self.__reference_value_upper = self.getN()+2-1.908
        else:
            self.__reference_value_upper = self.getN() + 2 - Util.robbins_problem_lower_boundary(self.getN())
        # self.__reference_value_upper = self.getN()+2-Util.robbins_problem_experimental_upper_boundary(self.getN())  # TODO: remember
        self.__reference_value_lower = self.getN()+2-Util.robbins_problem_lower_boundary(self.getN())  # explicit threshhold function

    def get_reference_value(self):
        return self.__reference_value_lower, self.__reference_value_upper



