import copy

import numpy as np
import pytest
import torch
import Util
from RobbinsModel import RobbinsModel

from ModelDefinitions import sigma_dict, payoff_dict, mu_dict
import scipy.stats
import random
from AbstractMathematicalModel import AbstractMathematicalModel


# important note: Ich maximiere den Rang statt ihn zu minimieren. Es gibt N+1 Ränge
class Shortened_RobbinsModel(RobbinsModel):  # TODO: might be a bad choice
    def __init__(self, N):
        super().__init__(N)

    def update_parameter_string(self, main_pc):
        parameter_string = "Shortened Robbins Model mit unterem Referenzwert: ", round(self.__reference_value_upper, 3), "oberem Referenzwert: ", round(self.__reference_value_lower, 3), "N: ", self._N + 1,\
                           "auf dem " + main_pc

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)

        self.parameter_string = parameter_string + "\n"

    # antithetic is ignored
    def generate_paths(self, L, antithetic=None, sorted=False, N=None):
        """
        for l in range(L):
            out.append(np.random.uniform(low=0.0, high=1.0, size=None))
            # if l % 100000 == 0:
            #     print(l)
        """

        if N is None:
            N = self._N

        dim = self.getpath_dim()

        x = np.random.uniform(low=0.0, high=1.0, size=(L, N+1)).tolist()
        # dim = np.amin([dim, (N+1)*np.ones_like(dim)], axis=0)

        y = []
        for l in range(L):
            y.append([])
            y[l].append([x[l][0]])
            for n in range(1, N+1):
                y[l].append(copy.deepcopy(y[l][n - 1]))
                y[l][n].append(x[l][n])

        Util.sort_lists_inplace_except_last_one(y)
        for l in range(L):
            for n in range(0, N):  # very important that I modify (0, N) so that the last list isn't modified
                y[l][n] = y[l][n][-dim[n]:]

        assert L == y.__len__()

        return y

    def get_time_partition(self, N=None, step_size=1):
        return range(1, self.getN()+1)

    def getT(self):
        return self._N

    def getN(self):
        return self._N

    def getK(self):
        return self._K

    def getprocess_dim(self):
        return 1

    # hat länge n statt n+1 weil der letzte Eintrag nicht gebraucht wird und werden darf
    def getpath_dim(self):
        """
        n = self._N
        out = np.ones(n)
        x_offset = 2
        for x in range(1, n-x_offset):
            out[x] = np.floor(n-np.nanmax([(n-2)*np.log(n-x_offset-x)/np.log(n-x_offset), n-x-1]))
        for x in range(n-x_offset, n):
            out[x] = x+1

        assert True
        return out.astype('int')
        """
        n = self._N
        out = np.ones(n)
        x_offset = 2
        for x in range(1, n - x_offset):
            out[x] = np.floor(n - np.nanmax([(n - 2) * np.log(2*n - x_offset - 2*x) / np.log(2*n - x_offset), n - x - 1]))
        for x in range(n - x_offset, n):
            out[x] = x + 1

        assert True
        return out.astype('int')

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
        self.__reference_value_upper = self.getN() + 2 - Util.robbins_problem_known_upper_boundary(self.getN())
        # self.__reference_value_upper = self.getN()+2-Util.robbins_problem_experimental_upper_boundary(self.getN())  # TODO: remember
        self.__reference_value_lower = self.getN()+2-Util.robbins_problem_lower_boundary(self.getN())  # explicit threshhold function

    def get_reference_value(self):
        return self.__reference_value_lower, self.__reference_value_upper

    def convert_Robbins_paths_to_shortened_Robbins_paths(self, lists):
        assert len(lists[0]) == self.getN()+1
        dim = self.getpath_dim()
        L = len(lists)
        out = Util.sort_lists_inplace_except_last_one(lists, in_place=False)
        for l in range(L):
            for n in range(0, self._N):
                out[l][n] = out[l][n][-dim[n]:]

        return out



