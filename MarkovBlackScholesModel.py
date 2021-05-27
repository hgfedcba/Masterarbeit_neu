import numpy as np
from ModelDefinitions import sigma_dict, payoff_dict, mu_dict
from AbstractMathematicalModel import AbstractMathematicalModel


# noinspection SpellCheckingInspection
class MarkovBlackScholesModel(AbstractMathematicalModel):
    def __init__(self, T, N, d, K, delta, mu, sigma, g, xi):
        self.__T = T  # Time Horizon
        self.__N = N
        self.__d = d  # Dimension of the underlying Problem
        self.__K = K
        self.__delta = delta
        self.__internal_mu = mu  # drift coefficient
        self.__internal_sigma = sigma  # standard deviation of returns
        self.__internal_g = g  # objective function
        self.__xi = xi  # Startwert
        self.__reference_value = -1
        self.t = self.get_time_partition(N)
        self.parameter_string = ""
        self.parameter_list = []

    def update_parameter_string(self):
        parameter_string = "reference_value: ", round(self.__reference_value, 3), "T: ", self.__T, "N: ", self.__N, "d: ", self.__d, "K: ", self.__K, "delta: ", self.__delta, "mu: ", mu_dict.get(self.__internal_mu), "sigma: ",\
                           sigma_dict.get(self.__internal_sigma), "g: ", payoff_dict.get(self.__internal_g), "xi: ", self.__xi

        parameter_string = ''.join(str(s) + " \t" for s in parameter_string)
        # parameter_string = mylog(parameter_string)
        self.parameter_string = parameter_string + "\n"

    def generate_paths(self, L, antithetic):
        bms = []
        out = []
        """
        for l in range(L):
            if not antithetic or l < L / 2:
                bms.append(self.generate_bm())
            elif l == L / 2:
                bms.extend([-item for item in bms])
            out.append(self.generate_path_from_bm(bms[l]))
        """
        for l in range(L):
            if not antithetic or l % 2 == 0:
                bms.append(self.generate_bm())
            else:
                bms.append(-bms[-1])
            out.append(self.generate_path_from_bm(bms[-1]))
            # if l % 100000 == 0:
            #     print(l)

        assert L == out.__len__()

        return out

    # Divide __T in Number steps. Skip every step_size step.
    def get_time_partition(self, Number, step_size=1):
        assert Number % step_size == 0  # TODO: I have to convert something here to int but am too lazy to find out what
        out = np.zeros(int(Number / step_size) + 1)

        for n in range(int(Number / step_size)):
            out[n + 1] = step_size * (n + 1) * self.__T / Number
            # assert out[n] != out[n + 1]

        return out

    def getT(self):
        return self.__T

    def getN(self):
        return self.__N

    def setN(self, N):
        self.__N = N

    def getK(self):
        return self.__K

    def getdelta(self):
        return self.__delta

    # single digit
    def getprocess_dim(self):
        return self.__d

    # vector
    def getpath_dim(self):
        return self.__d * np.ones(self.__N, dtype=np.int)

    def getxi(self):
        return self.__xi

    def getmu(self, x):
        self.assert_x_in_Rd(x, self.getprocess_dim())

        out = self.__internal_mu(x)

        self.assert_x_in_Rd(out, self.getprocess_dim())

        return out

    def getsigma(self, x):
        self.assert_x_in_Rd(x, self.getprocess_dim())

        out = self.__internal_sigma(x)

        if self.getprocess_dim() > 1:
            assert type(out).__module__ == 'numpy'
            assert out.shape[0] == self.getprocess_dim()
            assert out.shape[1] == self.getprocess_dim()
        else:
            assert isinstance(out, (int, float)) or out.size == 1

        return out

    def getg(self, t, x):
        assert 0 <= t <= self.getT()
        # disabled when i changed the d
        # self.assert_x_in_Rd(x, self.getpath_dim())

        out = self.__internal_g(t, x)

        # torch...
        # self.assert_x_in_Rd(out, 1)

        return out

    def setmu(self, mu):
        self.__internal_mu = mu

    def setsigma(self, sigma):
        self.__internal_sigma = sigma

    def set_reference_value(self, v):
        self.__reference_value = v

    def get_reference_value(self):
        return self.__reference_value
