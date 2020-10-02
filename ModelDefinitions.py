import numpy as np
import math


def add_sigma_c_x(sigma_constant):
    assert sigma_constant > 0

    def sigma(x):
        if type(x).__module__ != np.__name__:
            out = sigma_constant * x
        else:
            out = sigma_constant * (np.identity(x.shape[0]) * x)
        return out
    f = sigma
    sigmas.append(f)
    sigma_dict[f] = str(sigma_constant) + " * x"
    return f


sigmas = []
sigma_dict = {}


def add_mu_c_x(mu_constant, delta):
    def mu_c_x(x):
        out = (mu_constant - delta) * x
        return out
    f = mu_c_x
    mus.append(f)
    mu_dict[f] = str(mu_constant-delta) + " * x"
    return f


mus = []
mu_dict = {}


def add_american_put(d, K, r):
    def american_put(t_in, x):
        summands = []
        for j in range(d):
            summands.append(max(K - x[j].item(), 0))
            # sum += c - x[j]
        # return torch.exp(-r * t) * sum
        return math.exp(-r * t_in) * sum(summands)
    f = american_put
    payoffs.append(f)
    payoff_dict[f] = "american put"
    return f


def add_bermudan_max_call(r, K):
    def bermudan_max_call(t_in, x):
        return math.exp(-r * t_in) * max(max(x) - K, 0)
    f = bermudan_max_call
    payoffs.append(f)
    payoff_dict[f] = "bermudan max call"
    return f


"""
def american_put_from_bm(t_in, x):
    summands = []
    for j in range(self.d):
        summands.append(x[j].item())
    return math.exp(-self.r * t_in) * max(self.K - self.xi * math.exp((self.r - 0.5 * self.sigma_constant ** 2) * t_in + self.sigma_constant / self.d ** 0.5 + sum(summands)), 0)
"""

payoffs = []
payoff_dict = {}


def binomial_trees(S0, r, sigma, T, N, K):
    delta_T = T / N

    alpha = math.exp(r * delta_T)
    beta_local = ((alpha ** -1) + alpha * math.exp(sigma ** 2 * delta_T)) / 2
    u = beta_local + (beta_local ** 2 - 1) ** 0.5
    d = u ** -1
    q = (alpha - d) / (u - d)
    assert 1 > q > 0

    S = np.ones((N + 1, N + 1)) * S0

    for i in range(1, N + 1):
        for j in range(i + 1):
            S[j][i] = S[0][0] * (u ** j) * (d ** (i - j))
            assert True

    V = np.ones((N + 1, N + 1))
    V_map = np.ones((N + 1, N + 1)) * -2
    for i in range(N, -1, -1):
        for j in range(i, -1, -1):
            if i == N:
                V[j][i] = max(K - S[j][i], 0)
            else:
                h1 = max(K - S[j][i], 0)
                h2 = alpha ** -1 * (q * V[j + 1][i + 1] + (1 - q) * V[j][i + 1])
                V[j][i] = max(h1, h2)

                V_map[j][i] = h1 > h2  # DEBUG: a one indicates exercising is good
    return V[0][0]  # tested
