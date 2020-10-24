import datetime
import time
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch

from Util import *


# TODO: durationgraph: summiere val_frequency viele graphen aufeinander

def create_graphics(Memory, ProminentResults, Model, Config, run_number):
    create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults)
    create_net_pdf(run_number)


def create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults):
    pdf = pdfp.PdfPages("Metrics" + str(run_number) + ".pdf")

    # Value over time Graph
    plot_number_paths = 1
    fig1 = plt.figure(plot_number_paths)
    x = range(0, Config.validation_frequency * (len(Memory.val_discrete_payoff_list)), Config.validation_frequency)
    x = np.array(x)
    draw_connected_points(x, Memory.val_continuous_payoff_list, plot_number_paths)
    draw_connected_points(x, Memory.val_discrete_payoff_list, plot_number_paths)
    draw_connected_points(x, Model.get_reference_value()*np.ones_like(x), plot_number_paths, 'black')
    plt.legend(["cont value", "disc value", "reference value"])
    # plt.axvline(ProminentResults.disc_best_result.m, color="orange")
    # plt.axvline(ProminentResults.cont_best_result.m, color="blue")
    xlabel('t', fontsize=16)
    ylabel('payoff', fontsize=16)
    grid(True)

    pdf.savefig(fig1)
    plt.close(fig1)

    # Duration over time graph
    plot_number_duration = 2
    fig2 = plt.figure(plot_number_duration)
    # t = np.array(range(iteration_number, step=Config.validation_frequency))
    t = []
    td = []
    vd = []
    tn = []

    for k in range(0, int(len(Memory.train_durations)/Config.validation_frequency)):
        t.append(Config.validation_frequency*k)
        td.append(sum(Memory.train_durations[Config.validation_frequency*k:Config.validation_frequency*(k+1)]))
        vd.append(sum(Memory.val_durations[k:(k+1)]))
        tn.append(sum(Memory.total_net_durations[Config.validation_frequency*k:Config.validation_frequency*(k+1)]))
    draw_connected_points(t, td, plot_number_duration)
    draw_connected_points(t, vd, plot_number_duration)
    draw_connected_points(t, tn, plot_number_duration)
    plt.legend(["train duration", "val duration", "time spend in net"])
    xlabel('t', fontsize=16)
    ylabel('time', fontsize=16)
    plt.title('time spend on ' + str(Config.validation_frequency) + ' iterations')
    grid(True)

    pdf.savefig(fig2)
    plt.close(fig2)

    # visualized stopping times graph

    pdf.close()


def create_net_pdf(iteration_number):
    assert True