import datetime
import time
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch

from Util import *


# TODO: durationgraph: summiere val_frequency viele graphen aufeinander

def create_graphics(Memory, ProminentResults, Model, Config, iteration_number):
    create_metrics_pdf(iteration_number, Memory, Config, Model, ProminentResults)
    create_net_pdf(iteration_number)


def create_metrics_pdf(iteration_number, Memory, Config, Model, ProminentResults):
    pdf = pdfp.PdfPages("Metrics" + str(iteration_number) + ".pdf")

    # Value over time Graph
    plot_number_paths = 1
    fig1 = plt.figure(plot_number_paths)
    x = range(0, Config.validation_frequency * (len(Memory.val_discrete_value_list)), Config.validation_frequency)
    x = np.array(x)
    draw_connected_points(x, Memory.val_continuous_value_list, plot_number_paths)
    draw_connected_points(x, Memory.val_discrete_value_list, plot_number_paths)
    draw_connected_points(x, Model.get_reference_value()*np.ones_like(x), plot_number_paths, 'black')
    plt.legend(["cont value", "disc value", "reference value"])
    # TODO: vertical lines
    # plt.axvline(self.best_results.disc_best_result.m, color="red")
    # plt.axvline(self.best_results.cont_best_result.m, color="red")

    pdf.savefig(fig1)
    plt.close(fig1)

    # Duration over time graph

    # visualized stopping times graph

    pdf.close()


def create_net_pdf(iteration_number):
    assert True