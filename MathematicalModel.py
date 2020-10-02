import numpy as np
from ModelDefinitions import sigma_dict, payoff_dict, mu_dict


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
        self.parameter_string =""

    def update_parameter_string(self):
        parameter_string = "T: ", self.__T, "N: ", self.__N, "d: ", self.__d, "K: ", self.__K, "delta: ", self.__delta, "mu: ", mu_dict.get(self.__internal_mu), "sigma: ",\
                           sigma_dict.get(self.__internal_sigma), "g: ", payoff_dict.get(self.__internal_g), "xi: ", self.__xi, "reference_value: ", self.__reference_value

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)
        self.parameter_string = parameter_string + "\n"

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
