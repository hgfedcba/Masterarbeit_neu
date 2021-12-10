import copy

import numpy as np
import pytest
import torch

from ModelDefinitions import sigma_dict, payoff_dict, mu_dict
import scipy.stats
import random
import Util
from AbstractMathematicalModel import AbstractMathematicalModel


# important note: Ich maximiere den Rang statt ihn zu minimieren. Es gibt N+1 Ränge
class W_RobbinsModel(AbstractMathematicalModel):
    def __init__(self, N):
        self._N = N  # X_0,..., X_(N)
        self._reference_value = None
        self.t = self.get_time_partition()
        self.parameter_string = ""
        self.parameter_list = []
        self._K = 0  # solve this better. This exists since K is the offset towards the origin for the nets   (f.e. K=0.5 :P)

    def update_parameter_string(self, main_pc):
        parameter_string = "Robbins Model ohne Vergangenheit mit Referenzwert: ", round(self._reference_value, 3), "N: ", self._N + 1, "auf dem " + main_pc

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)

        self.parameter_string = parameter_string + "\n"

    # antithetic is ignored
    def generate_paths(self, L, antithetic=None, N=None):
        x = np.random.uniform(low=0.0, high=1.0, size=(L, 1, self._N + 1))
        return x

    def convert_Robbins_path_to_W_path(self, x):
        return np.array(x[-1])

    def convert_multiple_Robbins_paths_to_W_paths(self, x):
        l = len(x[0])
        return [np.array(y[-1]).reshape((1, l)) for y in x]

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
        return 1 * np.ones(self._N, dtype=np.int)

    def getg(self, t, x):
        # (zeitstetige) stoppzeit t, pfad x
        assert t.sum() == 1

        h = np.reshape(x, self.getN()+1)

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

    def calculate_payoffs(self, U, x, g, t, device=None):
        if isinstance(U, np.ndarray):
            return self.getg(U, x)
        assert torch.sum(torch.tensor(U)).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(torch.tensor(U)).item())

        # h = np.reshape(x, self.getN()+1)
        h = np.reshape(x, U.shape[0])

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

        h = torch.matmul(U, torch.tensor(z2, dtype=torch.float))

        # Schritt 2: Bilde das Skalarprodukt von t und z1
        return torch.matmul(U, torch.tensor(z2, dtype=torch.float))

    def set_reference_value(self):
        self._reference_value = self.getN() + 1 - Util.robbins_problem_lower_boundary(self.getN())

    def get_reference_value(self):
        return self._reference_value



