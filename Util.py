import scipy.stats
import numpy as np

def mylog(*argv, only_return=False):
    argv = list(argv)
    for s in range(len(argv)):
        if isinstance(argv[s], float):
            argv[s] = round(argv[s], 3)
    out = ''.join(str(s) + "\t" for s in argv)
    out += "\n"
    if not only_return:
        # TODO: log not defined
        log.info(out)
    return out

"""
def generate_bm(d, N, t):


def generate_path(bm, d, N, xi, t, mu, sigma):


def generate_antithetic_path(bm, d, N, xi, t, mu, sigma):
"""