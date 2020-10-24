import scipy.stats
import numpy as np
import time
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch


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


def draw_function(x, f, plot_number=0):
    if plot_number == 0:
        h = time.time()
        plot_number = int(h)
    plt.figure(plot_number)

    y = []
    for c in x:
        h = torch.tensor([c], dtype=torch.float32)
        y.append(f(h))
    plot(x, y)

    # TODO: Das gehört hier nicht hin
    """
    xlabel('x', fontsize=16)
    ylabel('f(x)', fontsize=16)
    plt.ylim([0, 1])
    grid(True)
    # show()
    # plt.close(fig)
    """

    return plot_number


def draw_connected_points(x, y, plot_number=0, color=None):
    if plot_number == 0:
        h = time.time()
        plot_number = int(h)
    plt.figure(plot_number)

    # for l in range(len(x)):
    #     plot(x, y[l].flatten())
    plot(x, y, color=color)
    """
    # TODO: Das gehört hier nicht hin
    xlabel('t', fontsize=16)
    ylabel('x', fontsize=16)
    grid(True)
    # show()
    # plt.close(fig)
    """

    return plot_number


#TODO: Recall the command for isolated points ist plt.scatter

"""
def generate_bm(d, N, t):


def generate_path(bm, d, N, xi, t, mu, sigma):


def generate_antithetic_path(bm, d, N, xi, t, mu, sigma):
"""