import math

from MarkovBlackScholesModel import MarkovBlackScholesModel
import numpy as np
from ModelDefinitions import payoff_dict, add_russian_option


class RussianOption(MarkovBlackScholesModel):
    def __init__(self, T, N, d, K, delta, mu, sigma, g, xi):
        super().__init__(T, N, d, K, delta, mu, sigma, g, xi)
        assert payoff_dict.get(self._internal_g) == "russian option"

    def update_parameter_string(self, main_pc):
        super().update_parameter_string(main_pc)

    @staticmethod
    def path_to_russian_path(p):
        if isinstance(p, np.ndarray):
            out = np.zeros((2, p.shape[1]))
            for k in range(p.shape[1]):
                out[0][k] = p[0][k]
                out[1][k] = max(out[1][k-1], p[0][k])
        else:
            for k in range(len(p)):
                p[k] = [p[k], max(p[k], p[k-1][1])]
            out = p
        return out

    def generate_paths(self, L, antithetic=True):
        out = super().generate_paths(L, antithetic)
        for k in range(len(out)):
            out[k] = self.path_to_russian_path(out[k])

        return out

    def getpath_dim(self):
        assert self._d == 1
        return 2 * np.ones(self._N, dtype=np.int)

    """
    def set_reference_value(self, r, sigma_constant):
        pass
        
        from scipy.integrate import quad, dblquad
        T = self._T
        lam = 0  # discount rate vs r interest rate   TODO: hä

        sigma = sigma_constant
        beta = r/sigma + sigma/2

        dblquad(lambda x, y: x * y, 0, 0.5, lambda x: 0, lambda x: 1 - 2 * x)

        def f(t, s, m):
            exponent = -(math.log(m**2/s)**2)/(2*sigma**2*t)+beta/sigma*math.log(s)-beta**2/2*t
            return 2/(sigma**3*math.sqrt(2*math.pi*t**3))*math.log(m**2/s)/(s*m)*math.exp(exponent)

        def F(t, x):
            return dblquad(lambda s, m: max(m, x)/s * f(t, s, m), 1, np.inf, lambda m: 0, lambda m: m)[0]

        def G(t, x, y):
            return dblquad(lambda s, m: max(m, x)/s * (max(m, x)/s >= y)*f(t, s, m), 1, np.inf, lambda m: 0, lambda m: m)[0]

        def b(t):
            return 0

        def V(t, x):
            return math.exp(-lam*(T-t)) * F(T-t, x) + (r+lam) * quad(lambda u: math.exp(-lam*u) * G(u, x, b(t+u)), 0, (T-t))
    """



