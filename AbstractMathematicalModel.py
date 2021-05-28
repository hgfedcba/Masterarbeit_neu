from abc import ABC, abstractmethod

import pytest
import scipy.stats
import numpy as np
import torch


class AbstractMathematicalModel(ABC):
    # general methods
    @abstractmethod
    def getprocess_dim(self):
        pass

    @abstractmethod
    def getpath_dim(self):
        pass

    @abstractmethod
    def getN(self):
        pass

    @abstractmethod
    def getg(self, t, x):
        pass

    @abstractmethod
    def update_parameter_string(self):
        pass

    @abstractmethod
    def generate_paths(self, L, antithetic):
        pass

    @abstractmethod
    def get_time_partition(self, N, step_size=1):
        pass

    def assert_x_in_Rd(self, x, d):
        if d > 1:
            assert type(x).__module__ == 'numpy'
            assert x.size == d
        else:
            assert isinstance(x, (int, float)) or x.size == 1

    # For the Black-Scholes-Models
    # noinspection PyUnresolvedReferences
    def generate_bm(self):
        # Ein Rückgabewert ist ein np.array der entsprechenden Länge, in dem die Werte über den gesamten sample path eingetragen sind
        out = np.zeros((self.getprocess_dim(), self.getN() + 1))
        for m in range(self.getprocess_dim()):
            for n in range(self.getN()):
                out[m, n + 1] = scipy.stats.norm.rvs(loc=out[m, n], scale=(self.t[n + 1] - self.t[n]) ** 0.5)

        return out

    # noinspection PyUnresolvedReferences
    def generate_path_from_bm(self, bm):
        out = np.zeros((self.getprocess_dim(), self.getN() + 1))
        out[:, 0] = self.getxi()
        for n in range(self.getN()):
            h = out[:, n]
            part2 = self.getmu(out[:, n]) * (self.t[n + 1] - self.t[n])
            part3 = self.getsigma(out[:, n]) @ (bm[:, n + 1] - bm[:, n])
            out[:, n + 1] = out[:, n] + part2 + part3
            # out[:, n + 1] = out[:, n] * (1 + part2 + part3)

        # return self.Sim_Paths_GeoBM(self.Model.getxi(), self.Model.getmu(1), self.Model.getsigma(1), self.Model.getT(), self.N)
        return out

    def calculate_payoffs(self, U, x, g, t):
        assert torch.sum(torch.tensor(U)).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(torch.tensor(U)).item())

        s = torch.zeros(1)
        for n in range(self.getN() + 1):
            h1 = U[n]
            h2 = g(t[n], x[:, n])
            s += U[n] * g(t[n], x[:, n])
        return s

    def getmu(self, x):
        pass

    def getsigma(self, x):
        pass

    def getxi(self):
        pass

