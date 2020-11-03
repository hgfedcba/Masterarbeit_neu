import datetime
import time
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch

from Util import *


def create_graphics(Memory, ProminentResults, Model, Config, run_number, val_paths, final_val_paths, NN):
    create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, val_paths, final_val_paths)
    create_net_pdf(run_number, Memory, Config, Model, ProminentResults, NN)


def create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, test_paths, val_paths):
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
    xlabel('iteration', fontsize=16)
    ylabel('payoff', fontsize=16)
    grid(True)
    # plt.yscale('log')

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

    # Ich lasse die letzten val_frequency iterationen absichtlich aus dem plot heraus da es nicht einfach ist sie richtig anzuzeigen
    for k in range(0, int(len(Memory.train_durations)/Config.validation_frequency)):
        iter.append(Config.validation_frequency*k)
        td.append(sum(Memory.train_durations[Config.validation_frequency*k:Config.validation_frequency*(k+1)]))
        vd.append(sum(Memory.val_durations[k:(k+1)]))
        tn.append(sum(Memory.total_net_durations[Config.validation_frequency*k:Config.validation_frequency*(k+1)]))
    draw_connected_points(iter, td, plot_number_duration)
    draw_connected_points(iter, vd, plot_number_duration)
    draw_connected_points(iter, tn, plot_number_duration)
    plt.legend(["train duration", "val duration", "time spend in net"])
    xlabel('iteration', fontsize=16)
    ylabel('time', fontsize=16)
    plt.title('time spend on ' + str(Config.validation_frequency) + ' iterations')
    grid(True)

    pdf.savefig(fig2)
    plt.close(fig2)

    plot_number_duration_2 = 25
    fig25 = plt.figure(plot_number_duration_2)

    td = []
    vd = []
    tn = []

    for k in range(0, int(len(Memory.train_durations)/Config.validation_frequency)):
        td.append(Memory.pretrain_duration + sum(Memory.train_durations[:Config.validation_frequency*(k)]))
        vd.append(Memory.pretrain_duration + sum(Memory.val_durations[:(k)]))
        tn.append(Memory.pretrain_duration + sum(Memory.total_net_durations[:Config.validation_frequency*(k)]))

    draw_connected_points(iter, td, plot_number_duration_2)
    draw_connected_points(iter, vd, plot_number_duration_2)
    draw_connected_points(iter, tn, plot_number_duration_2)
    draw_connected_points(iter, Memory.pretrain_duration * np.ones_like(iter), plot_number_duration_2)
    draw_connected_points(iter, (time.time() - Memory.start_time) * np.ones_like(iter), plot_number_duration_2)
    draw_connected_points(iter, (time.time() - Memory.start_time - Memory.final_val_duration) * np.ones_like(iter), plot_number_duration_2)
    plt.legend(["train duration", "val duration", "time spend in net", "pretrain end", "final val end", "final val start"])
    xlabel('iteration', fontsize=16)
    ylabel('time', fontsize=16)
    plt.ylim([0, (time.time() - Memory.start_time)*1.05])
    plt.title('cumulative time spend')
    grid(True)

    pdf.savefig(fig25)
    plt.close(fig25)

    if Model.getd() == 1:
        # visualized stopping times graph
        plot_number_paths = 3
        fig3 = plt.figure(plot_number_paths)
        create_paths_plot(test_paths, Model, ProminentResults.disc_best_result.test_stopping_times, plot_number_paths, "best disc", True)
        pdf.savefig(fig3)
        plt.close(fig3)

        plot_number_paths = 4
        fig4 = plt.figure(plot_number_paths)
        create_paths_plot(test_paths, Model, ProminentResults.cont_best_result.test_stopping_times, plot_number_paths, "best cont", True)
        pdf.savefig(fig4)
        plt.close(fig4)

        plot_number_paths = 5
        fig5 = plt.figure(plot_number_paths)
        create_paths_plot(test_paths, Model, ProminentResults.final_result.test_stopping_times, plot_number_paths, "final", True)
        pdf.savefig(fig5)
        plt.close(fig5)

        # visualized stopping times graph
        plot_number_paths = 6
        fig6 = plt.figure(plot_number_paths)
        create_paths_plot(val_paths, Model, ProminentResults.disc_best_result.val_stopping_times, plot_number_paths, "best disc", False)
        pdf.savefig(fig6)
        plt.close(fig6)

        plot_number_paths = 7
        fig7 = plt.figure(plot_number_paths)
        create_paths_plot(val_paths, Model, ProminentResults.cont_best_result.val_stopping_times, plot_number_paths, "best cont", False)
        pdf.savefig(fig7)
        plt.close(fig7)

        plot_number_paths = 8
        fig8 = plt.figure(plot_number_paths)
        create_paths_plot(val_paths, Model, ProminentResults.final_result.val_stopping_times, plot_number_paths, "final", False)
        pdf.savefig(fig8)
        plt.close(fig8)

    pdf.close()


def create_paths_plot(val_paths, Model, stopping_times, plot_number, title, on_testpaths):
    number_of_plots = min(len(val_paths), 64)
    t = np.asarray(range(0, Model.getN() + 1)) / Model.getN() * Model.getT()
    for k in range(number_of_plots):
        stop_point = np.argmax(stopping_times[k])
        stopped_path = val_paths[k][0][:stop_point + 1]
        draw_connected_points(t[:stop_point + 1], stopped_path, plot_number)
        plt.scatter(t[stop_point], stopped_path[stop_point], marker='o')

    xlabel('t', fontsize=16)
    ylabel('payoff', fontsize=16)
    if on_testpaths:
        plt.title('stopped paths on test set, ' + title)
    else:
        plt.title('stopped paths on validation set, ' + title)
    plt.ylim([20, 60])
    grid(True)


def create_net_pdf(run_number, Memory, Config, Model, ProminentResults, NN):
    # TODO: copy graph so i only use a copy when it was still open   ?????
    pdf = pdfp.PdfPages("net graphs " + str(run_number) + ".pdf")
    n_sample_points = 81  # 81 and half stepsize seems way more reasonable
    d = Model.getd()
    """
    x = np.ones((n_samples, d))
    for i in range(0, n_samples):
        x[i] = np.ones(d) * (Model.getK() + i - 20)
    """
    short = Config.x_plot_range_for_net_plot
    x = np.reshape(np.linspace(short[0], short[1], n_sample_points), (n_sample_points, 1)) * np.ones((1, d))

    NN = ProminentResults.disc_best_result.load_state_dict_into_given_net(NN)
    l = len(NN.u)
    for k in range(l):
        draw_function(x, NN.u[k], k+1)
    NN = ProminentResults.cont_best_result.load_state_dict_into_given_net(NN)
    for k in range(l):
        draw_function(x, NN.u[k], k+1)
    NN = ProminentResults.final_result.load_state_dict_into_given_net(NN)
    for k in range(l):
        draw_function(x, NN.u[k], k+1)

    for k in range(l):
        c_fig = plt.figure(k+1)
        if not Config.pretrain_func is False:
            draw_function(x, Config.pretrain_func, k+1, "black")
            plt.legend(["best disc result", "best cont result", "final result", "pretrain"])
        else:
            plt.legend(["best disc result", "best cont result", "final result"])
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
