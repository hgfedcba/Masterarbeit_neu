import numpy as np
from ModelDefinitions import sigma_dict, payoff_dict, mu_dict
from Util import mylog
import scipy.stats


# noinspection SpellCheckingInspection
class MathematicalModel:
    def __init__(self, T, N, d, K, delta, mu, sigma, g, xi):
        self.__T = T  # Time Horizon
        self.__N=N #
        self.__d = d  # Dimension of the underlying Problem
        self.__K=K
        self.__delta=delta
        self.__internal_mu = mu  # drift coefficient
        self.__internal_sigma = sigma  # standard deviation of returns
        self.__internal_g = g  # objective function
        self.__xi = xi  # Startwert
        self.__reference_value = -1
        self.t = self.get_time_partition(N)
        self.parameter_string = ""
        self.parameter_list = []

    def update_parameter_string(self):
        parameter_string = "T: ", self.__T, "N: ", self.__N, "d: ", self.__d, "K: ", self.__K, "delta: ", self.__delta, "mu: ", mu_dict.get(self.__internal_mu), "sigma: ",\
                           sigma_dict.get(self.__internal_sigma), "g: ", payoff_dict.get(self.__internal_g), "xi: ", self.__xi, "reference_value: ", round(self.__reference_value,3)

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)
        # parameter_string = mylog(parameter_string)
        self.parameter_string = parameter_string + "\n"

    def generate_bm(self):
        # Ein Rückgabewert ist ein np.array der entsprechenden Länge, in dem die Werte über den gesamten sample path eingetragen sind
        out = np.zeros((self.getd(), self.getN() + 1))
        for m in range(self.getd()):
            for n in range(self.getN()):
                out[m, n + 1] = scipy.stats.norm.rvs(loc=out[m, n], scale=(self.t[n + 1] - self.t[n]) ** 0.5)

        return out

    def generate_path_from_bm(self, bm):
        out = np.zeros((self.getd(), self.getN() + 1))
        out[:, 0] = self.getxi()
        for n in range(self.getN()):
            h = out[:, n]
            part2 = self.getmu(out[:, n]) * (self.t[n + 1] - self.t[n])
            part3 = self.getsigma(out[:, n]) @ (bm[:, n + 1] - bm[:, n])
            out[:, n + 1] = out[:, n] + part2 + part3
            # out[:, n + 1] = out[:, n] * (1 + part2 + part3)

        # return self.Sim_Paths_GeoBM(self.Model.getxi(), self.Model.getmu(1), self.Model.getsigma(1), self.Model.getT(), self.N)
        return out

    def generate_paths(self, number, antithetic):
        bms = []
        out = []
        L = number
        for l in range(L):
            if not antithetic or l < L / 2:
                bms.append(self.generate_bm())
            elif l == L / 2:
                bms.extend([-item for item in bms])
            out.append(self.generate_path_from_bm(bms[l]))

        assert L == out.__len__()

        return out

    def getT(self):
        return self.__T

    # TODO: WRONG!

    def get_time_partition(self, N, step_size=1):
        out = np.zeros(int(N / step_size) + 1)

        for n in range(int(N / step_size)):
            out[n + 1] = step_size * (n + 1) * self.__T / N
            # assert out[n] != out[n + 1]

        return out

    def getN(self):
        return self.__N

    def getK(self):
        return self.__K

    def getdelta(self):
        return self.__delta

    def getd(self):
        return self.__d

    def getxi(self):
        return self.__xi

    def getmu(self, x):
        self.assert_x_in_Rd(x, self.getd())

        out = self.__internal_mu(x)

        self.assert_x_in_Rd(out, self.getd())

        return out

    def getsigma(self, x):
        self.assert_x_in_Rd(x, self.getd())

        out = self.__internal_sigma(x)

        if self.getd() > 1:
            assert type(out).__module__ == 'numpy'
            assert out.shape[0] == self.getd()
            assert out.shape[1] == self.getd()
        else:
            assert isinstance(out, (int, float)) or out.size == 1

        return out

    def getg(self, t, x):
        assert 0 <= t <= self.getT()
        self.assert_x_in_Rd(x, self.getd())

        out = self.__internal_g(t, x)

        # torch...
        # self.assert_x_in_Rd(out, 1)

        return out

    def setmu(self, mu):
        self.__internal_mu = mu

    def setsigma(self, sigma):
        self.__internal_sigma = sigma

    def assert_x_in_Rd(self, x, d):
        if d > 1:
            assert type(x).__module__ == 'numpy'
            assert x.size == d
        else:
            assert isinstance(x, (int, float)) or x.size == 1

    def set_reference_value(self, v):
        self.__reference_value = v

    def get_reference_value(self):
        return self.__reference_value
