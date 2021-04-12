import datetime
import time
import numpy as np
from pylab import plot, show, grid, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfp
import torch

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from Util import *


def create_graphics(Memory, ProminentResults, Model, Config, run_number, val_paths, final_val_paths, NN):
    create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, val_paths, final_val_paths)
    create_net_pdf(run_number, Memory, Config, Model, ProminentResults, NN)


def create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, test_paths, val_paths):
    pdf = pdfp.PdfPages("Metrics" + str(run_number) + ".pdf")

    # Value over time Graph
    plot_number_paths = 1
    fig1 = plt.figure(plot_number_paths)
    x = range(0, Config.validation_frequency * (len(Memory.val_discrete_value_list)), Config.validation_frequency)
    x = np.array(x)
    draw_connected_points(x, Memory.val_continuous_value_list, plot_number_paths)
    draw_connected_points(x, Memory.val_discrete_value_list, plot_number_paths)
    draw_connected_points(x, Model.get_reference_value()*np.ones_like(x), plot_number_paths, 'black')
    plt.legend(["cont value", "disc value", "reference value"])
    # plt.axvline(ProminentResults.disc_best_result.m, color="orange")
    # plt.axvline(ProminentResults.cont_best_result.m, color="blue")
    xlabel('iteration', fontsize=16)
    ylabel('value', fontsize=16)
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
    plt.legend(["train duration", "test duration", "time spend in net"])
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
    plt.legend(["train duration", "test duration", "time spend in net", "pretrain end", "final val end", "final val start"])
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
    ylabel('value', fontsize=16)
    if on_testpaths:
        plt.title('stopped paths on test set, ' + title)
    else:
        plt.title('stopped paths on validation set, ' + title)
    plt.ylim([20, 60])
    grid(True)


def create_net_pdf(run_number, Memory, Config, Model, ProminentResults, NN):
    pdf = pdfp.PdfPages("net graphs " + str(run_number) + ".pdf")
    n_sample_points = 86  # 81 and half stepsize seems way more reasonable
    d = Model.getd()

    if Model.getd() == 1 and Config.algorithm == 2:
        # new
        fig = plt.figure()
        # TODO: zeile drunter entkommentieren
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        """
        # Make data.
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.sin(R)
        """
        X = NN.K + np.arange(0, Model.getT(), Model.getT()*1.0 / n_sample_points)
        short = Config.x_plot_range_for_net_plot
        Y = np.arange(short[0], short[1], (short[1] - short[0])*1.0 / n_sample_points)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((n_sample_points, n_sample_points))
        for i in range(n_sample_points):
            for j in range(n_sample_points):
                into = (X[i][j], Y[i][j])
                h = torch.tensor(into, dtype=torch.float32, requires_grad=False)
                Z[i][j] = NN.u[0](h)
        # Plot the surface
        surf = ax.plot_surface(X-NN.K, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel("n")
        ax.set_ylabel("X")
        ax.set_zlabel("$u_n(X)$")
        ax.zaxis.set_rotate_label(False)

        # Customize the z axis
        ax.set_zlim(0.0, 1.0)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.view_init(30, 40)
        pdf.savefig(fig)
        """
        # rotate the axes and update
        for angle in range(0, 720, 5):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.5)

        fig.save("animation " + str(run_number) + ".gif", writer='imagemagick', fps=30)
        """
        plt.close(fig)

    """
    x = np.ones((n_samples, d))
    for i in range(0, n_samples):
        x[i] = np.ones(d) * (Model.getK() + i - 20)
    """
    short = Config.x_plot_range_for_net_plot
    x = np.reshape(np.linspace(short[0], short[1], n_sample_points), (n_sample_points, 1)) * np.ones((1, d))

    if Model.getd() == 2 and Config.algorithm == 2:
        for k in range(NN.N):
            fig_k = plt.figure(k)
            ax = fig_k.gca(projection='3d')
            """
            # Make data.
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X ** 2 + Y ** 2)
            Z = np.sin(R)
            """
            short = Config.x_plot_range_for_net_plot
            Y = np.arange(short[0], short[1], (short[1] - short[0]) * 1.0 / n_sample_points)

            X, Y = np.meshgrid(Y, Y)
            Z = np.zeros((n_sample_points, n_sample_points))
            for i in range(n_sample_points):
                for j in range(n_sample_points):
                    into = (k+NN.K, X[i][j], Y[i][j])
                    h = torch.tensor(into, dtype=torch.float32, requires_grad=False)
                    Z[i][j] = NN.u[0](h)
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel('u_%s' % k)
            ax.zaxis.set_rotate_label(False)

            # Customize the z axis
            ax.set_zlim(0.0, 1.0)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            # Add a color bar which maps values to colors
            fig_k.colorbar(surf, shrink=0.5, aspect=5)

            ax.view_init(30, Config.angle_for_net_plot)
            pdf.savefig(fig_k)
            """
            # rotate the axes and update
            for angle in range(0, 720, 5):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(.5)
    
            fig.save("animation " + str(run_number) + ".gif", writer='imagemagick', fps=30)
            """
            plt.close(fig_k)

    def get_net(u, k):
        if Config.algorithm == 0:
            return u[k]
        elif Config.algorithm == 2:
            return u[0]

    # TODO: recall: it is vitaly important that plot_number=k+1+NN.k
    l = NN.N
    NN = ProminentResults.disc_best_result.load_state_dict_into_given_net(NN)
    for k in range(l):
        draw_function(x, get_net(NN.u, k), plot_number=k+1+NN.K, algorithm=Config.algorithm)
    NN = ProminentResults.cont_best_result.load_state_dict_into_given_net(NN)
    for k in range(l):
        draw_function(x, get_net(NN.u, k), plot_number=k+1+NN.K, algorithm=Config.algorithm)
    NN = ProminentResults.final_result.load_state_dict_into_given_net(NN)
    for k in range(l):
        draw_function(x, get_net(NN.u, k), plot_number=k+1+NN.K, algorithm=Config.algorithm)

    for k in range(l):
        c_fig = plt.figure(k+1+NN.K)
        if Config.do_pretrain:
            draw_function(x, Config.pretrain_func, k+1+NN.K, "black")
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
