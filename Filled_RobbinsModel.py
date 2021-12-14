import copy

import numpy as np
import pytest
import torch
import Util

from ModelDefinitions import sigma_dict, payoff_dict, mu_dict
import scipy.stats
import random
from AbstractMathematicalModel import AbstractMathematicalModel


# important note: Ich maximiere den Rang statt ihn zu minimieren. Es gibt N+1 Ränge
class Filled_RobbinsModel(AbstractMathematicalModel):  # die pfade werden voene mit 0en aufgefüllt und sind sortiert
    def __init__(self, N):
        self._N = N  # X_0,..., X_(N)
        self._reference_value_upper_V = None
        self._reference_value_lower_W = None
        self.t = self.get_time_partition()
        self.parameter_string = ""
        self.parameter_list = []
        self._K = 0  # solve this better. This exists since K is the offset towards the origin for the nets   (f.e. K=0.5 :P)

    def update_parameter_string(self, main_pc):
        parameter_string = "Filled Robbins Model mit unterem Referenzwert für V: ", round(self._reference_value_upper_V, 3), "oberem Referenzwert für W: ", round(self._reference_value_lower_W, 3), "N: ", self._N + 1,\
                           "auf dem " + main_pc

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)

        self.parameter_string = parameter_string + "\n"

    def sort_np_paths_list(self, paths):
        for k in range(len(paths)):
            paths[k][:-1, :-1] = np.sort(paths[k][:-1, :-1], axis=0)
        return paths

    def convert_Robbins_to_Filled(self, list):
        y = []
        N = len(list[0])-1
        for i in range(len(list)):
            y.append(np.zeros([N + 1, N + 1]))
            for j in range(N+1):
                h1 = y[i][j, -j - 1:]
                h2 = list[i][j]
                y[i][-j - 1:, j] = np.array(list[i][j])
                assert True

        y = self.sort_np_paths_list(y)

        return y

    # antithetic is ignored
    def generate_paths(self, L, antithetic=None, sorted=False, N=None):
        if N is None:
            N = self._N

        x = np.random.uniform(low=0.0, high=1.0, size=(L, N+1))

        y = []
        for i in range(L):
            y.append(np.zeros([N+1, N+1]))
            for j in range(N+1):
                h1 = y[i][j, -j-1:]
                h2 = x[i, :j+1]
                y[i][-j-1:, j] = x[i, :j+1]

        y = self.sort_np_paths_list(y)

        return y

    def get_time_partition(self, N=None, step_size=1):
        return range(1, self._N + 1)

    def getT(self):
        return self._N

    def getN(self):
        return self._N

    def getK(self):
        return self._K

    def getprocess_dim(self):
        return 1

    def getpath_dim(self):
        return (self._N + 1) * np.ones(self._N, dtype=np.int)  # TODO: why N and not N+1?

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

    def calculate_payoffs(self, U, x, g, t, device="cpu"):
        if isinstance(U, np.ndarray):
            return self.getg(U, x)
        assert torch.sum(torch.tensor(U)).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(torch.tensor(U)).item())

        h = self.convert_NN_path_to_mathematical_path(x)

        # Schritt 1: Ersetze in x jeden Wert mit dem entsprechenden Rang
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
        h = torch.matmul(U, torch.tensor(z2, dtype=torch.float, device=device))  # TODO: Hier wird ein vektor von der cpu auf die gpu geladen -> Zeitproblem

        return torch.matmul(U, torch.tensor(z2, dtype=torch.float, device=device))

    def set_reference_value(self):
        # 1.908 ist die beste bekannte untere Grenze für V und da die V_n monoton wachsend sind ist es auch eine untere Grenze für V_n
        self._reference_value_upper_V = self.getN() + 2 - Util.robbins_problem_known_upper_boundary_of_V(self.getN())
        # self.__reference_value_upper = self.getN()+2-Util.robbins_problem_experimental_upper_boundary(self.getN())  # TODO: remember
        self._reference_value_lower_W = self.getN() + 2 - Util.robbins_problem_lower_boundary_of_W(self.getN())  # explicit threshhold function

    def get_reference_value(self):
        return self._reference_value_lower_W, self._reference_value_upper_V



