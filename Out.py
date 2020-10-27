import datetime
import time
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch

from Util import *


# TODO: durationgraph: summiere val_frequency viele graphen aufeinander

def create_graphics(Memory, ProminentResults, Model, Config, run_number, val_paths):
    # TODO: Loss mit logarithmisher skala
    create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, val_paths)
    create_net_pdf(run_number, Memory, Config, Model, ProminentResults)


def create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, val_paths):
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
    plt.yscale('log')

    pdf.savefig(fig1)
    plt.close(fig1)

    # Duration over time graph
    plot_number_duration = 2
    fig2 = plt.figure(plot_number_duration)
    # t = np.array(range(iteration_number, step=Config.validation_frequency))
    iter = []
    td = []
    vd = []
    tn = []

    for k in range(0, int(len(Memory.train_durations)/Config.validation_frequency)):
        iter.append(Config.validation_frequency*k)
        td.append(sum(Memory.train_durations[Config.validation_frequency*k:Config.validation_frequency*(k+1)]))
        vd.append(sum(Memory.val_durations[k:(k+1)]))
        tn.append(sum(Memory.total_net_durations[Config.validation_frequency*k:Config.validation_frequency*(k+1)]))
    draw_connected_points(iter, td, plot_number_duration)
    draw_connected_points(iter, vd, plot_number_duration)
    draw_connected_points(iter, tn, plot_number_duration)
    plt.legend(["train duration", "val duration", "time spend in net"])
    xlabel('t', fontsize=16)
    ylabel('time', fontsize=16)
    plt.title('time spend on ' + str(Config.validation_frequency) + ' iterations')
    grid(True)

    pdf.savefig(fig2)
    plt.close(fig2)

    # visualized stopping times graph
    plot_number_paths = 3
    fig3 = plt.figure(plot_number_paths)
    number_of_plots = len(val_paths)
    t = np.asarray(range(0, Model.getN()+1))/Model.getN()*Model.getT()
    for k in range(number_of_plots):
        stop_point = np.argmax(ProminentResults.disc_best_result.stopping_times[k])
        stopped_path = val_paths[k][0][:stop_point]
        draw_connected_points(t[:stop_point], stopped_path, plot_number_paths)
        plt.scatter(t[stop_point-1], stopped_path[stop_point-1], marker='o')

    xlabel('t', fontsize=16)
    ylabel('payoff', fontsize=16)
    plt.title('stopped paths on validation set')
    plt.ylim([20, 60])
    grid(True)

    pdf.savefig(fig3)
    plt.close(fig3)
    pdf.close()


def create_net_pdf(run_number, Memory, Config, Model, ProminentResults):
    # TODO: copy graph so i only use a copy when it was still open   ?????
    pdf = pdfp.PdfPages("net graphs " + str(run_number) + ".pdf")
    n_samples = 41  # 81 and half stepsize seems way more reasonable
    d = Model.getd()
    x = np.ones((n_samples, d))
    for i in range(0, n_samples):
        x[i] = np.ones(d) * (Model.getK() + i - 20)
    NN = ProminentResults.disc_best_result.NN
    l = len(NN.u)
    for k in range(l):
        c_fig = plt.figure(1)
        draw_function(x, NN.u[k], 1)
        if d == 1:
            label = "x"
        else:
            label = "(x)^" + str(d)
        xlabel(label, fontsize=16)
        ylabel('u_%s' % k, fontsize=16)
        plt.ylim([0, 1])
        grid(True)
        pdf.savefig(c_fig)
        plt.close(c_fig)

    pdf.close()
