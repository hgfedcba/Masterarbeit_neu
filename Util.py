import scipy.stats
import numpy as np
import time
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch

"""
def draw_net(x, f, plot_number=0, color=None, algorithm=0):
    if algorithm == 0:
        return draw_function(x, f[plot_number-1], plot_number=plot_number, color=color, algorithm=algorithm)
    elif algorithm == 2:
        return draw_function(x, f[0], plot_number=plot_number, color=color, algorithm=algorithm)
"""


def draw_function(x, f, plot_number=0, color=None, algorithm=0):
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
        if algorithm == 0:
            h = torch.tensor([c], dtype=torch.float32, requires_grad=False)
        elif algorithm == 2 or algorithm == 3:
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


def draw_connected_points(x, y, plot_number=0, color=None):
    if plot_number == 0:
        h = time.time()
        plot_number = int(h)
    plt.figure(plot_number)

    # for l in range(len(x)):
    #     plot(x, y[l].flatten())
    plot(x, y, color=color)

    return plot_number
