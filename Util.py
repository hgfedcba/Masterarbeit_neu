import scipy.stats
import numpy as np
import time
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch


def sort_list_inplace(list, in_place=True, N=None):
    if in_place:
        out = list
    else:
        out = list.copy()
    if N is None:
        N = len(out[0])
    for k in range(len(out)):
        for l in range(N - 1):
            h = sorted(out[k][l][0:-1])
            h.append(out[k][l][-1])
            out[k][l] = h

    return out


def fake_net(x):
    return 0


def real_fake_net(j, N):
    def f(x):
        return x[j] > (N-j-1)/(N-j)
    return f


def robbins_problem_lower_boundary(n):
    n = n+1
    if n > 12:
        return 1 + 4*(n-1)/(3*(n+1))
    elif n == 1:
        return 1
    elif n == 2:
        return 1.25
    elif n == 3:
        return 1.4009
    elif n == 4:
        return 1.5606
    elif n == 5:
        return 1.5861
    elif n == 6:
        return 1.6490
    elif n == 7:
        return 1.7002
    elif n == 8:
        return 1.7430
    elif n == 9:
        return 1.7794
    elif n == 10:
        return 1.8109
    elif n == 11:
        return 1.8384
    elif n == 12:
        return 1.8627


# TODO: recall this conjecture is in the paper
def robbins_problem_experimental_upper_boundary(n):
    out = 1.908
    for k in range(1, n-4):
        out += 0.048*0.615**k
    return out


def robbins_problem_known_upper_boundary(n):

    return 1.908


def force_5_decimal(d):
    return f'{d:.5f}'


def mylog(*argv):
    argv = list(argv)
    for s in range(len(argv)):
        if isinstance(argv[s], float):
            argv[s] = round(argv[s], 3)
    out = ''.join(str(s) + "\t" for s in argv)
    out += "\n"
    return out


"""
def draw_net(x, f, plot_number=0, color=None, algorithm=0):
    if algorithm == 0:
        return draw_function(x, f[plot_number-1], plot_number=plot_number, color=color, algorithm=algorithm)
    elif algorithm == 2:
        return draw_function(x, f[0], plot_number=plot_number, color=color, algorithm=algorithm)
"""


def draw_function(x, f, plot_number=0, color=None, algorithm=False):
    """
    if plot_number == 0:
        h = time.time()
        plot_number = int(h)
    """
    plt.figure(plot_number)

    y = []
    actual_x = []
    for c in x:
        h = None
        if not algorithm:
            h = torch.tensor([c], dtype=torch.float32, requires_grad=False)
        else:
            h = torch.tensor([np.append(plot_number-1, c)], dtype=torch.float32, requires_grad=False)
        y.append(f(h))
        if len(c) > 1:
            actual_x.append(c[0])
        else:
            actual_x.append(c)
    # TODO: very fucking cool [ys.numpy() for ys in y]
    z = np.array([ys.detach().numpy()[0] for ys in y])

    plot(actual_x, z, color=color)
    return plot_number


def draw_connected_points(x, y, plot_number=0, color=None, do_scatter=False, line_style='-'):
    if plot_number == 0:
        h = time.time()
        plot_number = int(h)
    plt.figure(plot_number)

    # for l in range(len(x)):
    #     plot(x, y[l].flatten())
    if do_scatter:
        plt.scatter(x, y, color=color)
    else:
        plot(x, y, line_style, color=color)

    return plot_number
