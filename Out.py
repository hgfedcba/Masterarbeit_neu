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

import RussianOption
from RobbinsModel import RobbinsModel

from Util import *
import statistics


def average_value_stopped_at(final_result, Model, paths):
    c = []
    v = []
    stopping_times = Model.convert_vector_stopping_times_to_int(final_result.val_stopping_times)
    for n in range(Model.getN() + 1):
        stop = []
        no_stop = []
        if isinstance(Model, RobbinsModel):
            for k in range(len(paths)):
                if stopping_times[k] == n:
                    stop.append(paths[k][n][0])
                elif stopping_times[k] > n:
                    no_stop.append(paths[k][n][0])
        else:
            for k in range(paths.shape[0]):
                if stopping_times[k] == n:
                    stop.append(paths[k][0][n])
                elif stopping_times[k] > n:
                    no_stop.append(paths[k][0][n])

        if not stop:
            c.append('red')
            v.append(0)
        else:
            h = len(stop)
            if isinstance(Model, RobbinsModel):
                if h > len(paths) / Model.getN():
                    c.append('green')
                elif h > len(paths) / Model.getN() / 3:
                    c.append('yellow')
                else:
                    c.append('blue')
            else:
                if h > paths.shape[0] / Model.getN():
                    c.append('green')
                elif h > paths.shape[0] / Model.getN() / 2:
                    c.append('yellow')
                else:
                    c.append('blue')
            v.append(np.mean(stop))
    return c, v


def create_graphics(Memory, ProminentResults, Model, Config, run_number, val_paths, final_val_paths, NN):
    create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, val_paths, final_val_paths)
    create_net_pdf(run_number, Memory, Config, Model, ProminentResults, NN)
    plt.close('all')


def create_metrics_pdf(run_number, Memory, Config, Model, ProminentResults, test_paths, val_paths):
    pdf = pdfp.PdfPages("Metrics" + str(run_number) + ".pdf")

    # Value over time Graph
    plot_number_value = 1
    fig1 = plt.figure(plot_number_value)

    # plt.axvline(ProminentResults.disc_best_result.m, color="orange")
    # plt.axvline(ProminentResults.cont_best_result.m, color="blue")
    plt.title("value on test set over train iterations")
    xlabel('iteration', fontsize=16)
    ylabel('value', fontsize=16)
    grid(True)
    # plt.yscale('log')

    if Config.algorithm == 10:
        x = range(0, len(Memory.val_discrete_value_list))
        xlabel('nets trained', fontsize=16)
    else:
        x = range(0, Config.validation_frequency * (len(Memory.val_discrete_value_list)), Config.validation_frequency)
    x = np.array(x)
    draw_connected_points(x, Memory.val_continuous_value_list, plot_number_value)
    draw_connected_points(x, Memory.val_discrete_value_list, plot_number_value)

    r_value = Model.get_reference_value()
    if r_value == -1:
        plt.legend(["cont value", "disc value"])
    elif isinstance(r_value, float):
        draw_connected_points(x, r_value*np.ones_like(x), plot_number_value, 'black')
        plt.legend(["cont value", "disc value", "reference value"])
    else:
        draw_connected_points(x, r_value[0] * np.ones_like(x), plot_number_value, 'black')
        draw_connected_points(x, r_value[1] * np.ones_like(x), plot_number_value, 'black')
        plt.legend(["cont value", "disc value", "reference upper", "reference lower"])

    pdf.savefig(fig1)
    plt.close(fig1)

    # train Value over time Graph
    plot_number_train = 12
    fig12 = plt.figure(plot_number_train)
    x = range(0, len(Memory.average_train_payoffs))
    x = np.array(x)
    draw_connected_points(x, Memory.average_train_payoffs, plot_number_train)
    if r_value == -1:
        plt.legend(["cont value", "disc value"])
    elif isinstance(r_value, float):
        draw_connected_points(x, r_value * np.ones_like(x), plot_number_train, 'black')
        plt.legend(["train values", "reference value"])
    else:
        draw_connected_points(x, r_value[0] * np.ones_like(x), plot_number_train, 'black')
        draw_connected_points(x, r_value[1] * np.ones_like(x), plot_number_train, 'black')
        plt.legend(["train values", "reference upper", "reference lower"])
    # plt.axvline(ProminentResults.disc_best_result.m, color="orange")
    # plt.axvline(ProminentResults.cont_best_result.m, color="blue")
    plt.title("value on train batch each iteration")
    xlabel('iteration', fontsize=16)
    ylabel('value', fontsize=16)
    grid(True)
    # plt.yscale('log')

    pdf.savefig(fig12)
    plt.close(fig12)

    # average time of stopping
    plot_number_stopping_time = 13
    fig13 = plt.figure(plot_number_stopping_time)
    x = range(0, Config.validation_frequency * (len(Memory.average_test_stopping_time)), Config.validation_frequency)
    x = np.array(x)
    draw_connected_points(x, Memory.average_test_stopping_time, plot_number_stopping_time)
    plt.legend(["average time of stopping"])
    plt.title("average time we stop on test set")
    plt.ylim([0, Model.getN()])
    # plt.axvline(ProminentResults.disc_best_result.m, color="orange")
    # plt.axvline(ProminentResults.cont_best_result.m, color="blue")
    xlabel('iteration', fontsize=16)
    ylabel('time', fontsize=16)
    grid(True)
    # plt.yscale('log')

    pdf.savefig(fig13)
    plt.close(fig13)

    # bar graph when stopping happened on final val
    plot_number_final_stopping_time = 14
    fig14 = plt.figure(plot_number_final_stopping_time)
    x = range(0, Model.getN()+1)
    x = np.array(x)
    y = np.zeros(Model.getN()+1)
    for r in ProminentResults.final_result.val_stopping_times:
        y += r
    plt.bar(x, y)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Amount of Stops')
    plt.title('Exact times of stopping on the validation set')
    # plt.yscale('log')

    pdf.savefig(fig14)
    plt.close(fig14)

    if np.all(Model.getpath_dim() == np.ones_like(Model.getpath_dim())) or isinstance(Model, RobbinsModel):
        # bar graph of stopping boundary
        c, v = average_value_stopped_at(ProminentResults.final_result, Model, val_paths)

        # mean
        plot_number_final_stopping_boundary_mean = 15
        fig15 = plt.figure(plot_number_final_stopping_boundary_mean)
        plt.bar(x[:len(v)], v, color=c)
        plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('average value')
        plt.title('avg value stopped at on val set at each timestep')
        # plt.yscale('log')

        pdf.savefig(fig15)
        plt.close(fig15)

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
    # make this a pie chart     problem mit time spent in net
    plt.title('cumulative time spend')
    grid(True)

    pdf.savefig(fig25)
    plt.close(fig25)
    '''
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'pretrain', 'train', 'test', 'validation'
    sizes = [15, 30, 45, 10]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    '''
    path_dim = Model.getpath_dim()

    if np.allclose(path_dim, 1 * np.ones_like(path_dim)) or isinstance(Model, RussianOption.RussianOption):
        # visualized stopping times graph
        plot_number_paths = 3
        fig3 = plt.figure(plot_number_paths)
        create_paths_plot(test_paths, Model, Config, ProminentResults.disc_best_result.test_stopping_times, plot_number_paths, "best disc", True)
        pdf.savefig(fig3)
        plt.close(fig3)

        plot_number_paths = 4
        fig4 = plt.figure(plot_number_paths)
        create_paths_plot(test_paths, Model, Config, ProminentResults.cont_best_result.test_stopping_times, plot_number_paths, "best cont", True)
        pdf.savefig(fig4)
        plt.close(fig4)

        plot_number_paths = 5
        fig5 = plt.figure(plot_number_paths)
        create_paths_plot(test_paths, Model, Config, ProminentResults.final_result.test_stopping_times, plot_number_paths, "final", True)
        pdf.savefig(fig5)
        plt.close(fig5)

        # visualized stopping times graph
        plot_number_paths = 6
        fig6 = plt.figure(plot_number_paths)
        create_paths_plot(val_paths, Model, Config, ProminentResults.disc_best_result.val_stopping_times, plot_number_paths, "best disc", False)
        pdf.savefig(fig6)
        plt.close(fig6)

        plot_number_paths = 7
        fig7 = plt.figure(plot_number_paths)
        create_paths_plot(val_paths, Model, Config, ProminentResults.cont_best_result.val_stopping_times, plot_number_paths, "best cont", False)
        pdf.savefig(fig7)
        plt.close(fig7)

        plot_number_paths = 8
        fig8 = plt.figure(plot_number_paths)
        create_paths_plot(val_paths, Model, Config, ProminentResults.final_result.val_stopping_times, plot_number_paths, "final", False)
        pdf.savefig(fig8)
        plt.close(fig8)

    pdf.close()


def create_paths_plot(val_paths, Model, Config, stopping_times, plot_number, title, on_testpaths):
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
    plt.ylim(Config.x_plot_range_for_net_plot)
    grid(True)


def create_net_pdf(run_number, Memory, Config, Model, ProminentResults, NN):
    pdf = pdfp.PdfPages("net graphs " + str(run_number) + ".pdf")
    n_sample_points = 86  # 81 and half stepsize seems way more reasonable

    d = Model.getpath_dim()
    if np.allclose(d, np.ones_like(d) * d[0]):
        d = d[0]
    else:
        d = -1

    if d == 1 and NN.single_net_algorithm():
        # new
        fig = plt.figure()
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
        """
        # rotate the axes and update
        for angle in range(0, 720, 5):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.5)

        fig.save("animation " + str(run_number) + ".gif", writer='imagemagick', fps=30)
        """
        pdf.savefig(fig)
        plt.close(fig)

    """
    x = np.ones((n_samples, d))
    for i in range(0, n_samples):
        x[i] = np.ones(d) * (Model.getK() + i - 20)
    """
    short = Config.x_plot_range_for_net_plot

    if d == 2:
        for k in range(NN.N):
            fig_k = plt.figure(k)
            ax = fig_k.add_subplot(111, projection='3d')
            """
            # Make data.
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X ** 2 + Y ** 2)
            Z = np.sin(R)
            """
            Y = np.arange(short[0], short[1], (short[1] - short[0]) * 1.0 / n_sample_points)

            X, Y = np.meshgrid(Y, Y)
            Z = np.zeros((n_sample_points, n_sample_points))
            for i in range(n_sample_points):
                for j in range(n_sample_points):
                    if NN.single_net_algorithm():
                        into = (k+NN.K, X[i][j], Y[i][j])
                        h = torch.tensor(into, dtype=torch.float32, requires_grad=False)
                        Z[i][j] = NN.u[0](h)
                    else:
                        into = (X[i][j], Y[i][j])
                        h = torch.tensor(into, dtype=torch.float32, requires_grad=False)
                        Z[i][j] = NN.u[k](h)
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            # ax.set_zlabel('u_%s' % k)
            ax.zaxis.set_rotate_label(False)

            # Customize the z axis
            ax.set_zlim(0.0, 1.0)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.tick_params(axis='z', label1On=False)

            # Add a color bar which maps values to colors
            fig_k.colorbar(surf, shrink=0.5, aspect=5)

            # ax.view_init(30, Config.angle_for_net_plot)  # Seitenansicht
            ax.view_init(90, Config.angle_for_net_plot)  # Von oben

            """
            # rotate the axes and update
            for angle in range(0, 720, 5):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(.5)
    
            fig.save("animation " + str(run_number) + ".gif", writer='imagemagick', fps=30)
            """
            pdf.savefig(fig_k)
            plt.close(fig_k)

        # Plot paths
        fig_paths = plt.figure(4242)
        ax = fig_paths.add_subplot(111, projection='3d')

        number_of_paths_to_plot = 40

        some_ones = np.ones_like(Memory.val_paths[0][1]) / number_of_paths_to_plot

        for k in range(0, number_of_paths_to_plot):
            ax.plot(Memory.val_paths[k][0], Memory.val_paths[k][1], some_ones*k)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax.set_zlabel('u_%s' % k)
        ax.zaxis.set_rotate_label(False)

        # Customize the z axis
        ax.set_xlim(Config.x_plot_range_for_net_plot)
        ax.set_ylim(Config.x_plot_range_for_net_plot)
        ax.set_zlim(0.0, 1.0)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.tick_params(axis='z', label1On=False)

        # Add a color bar which maps values to colors
        fig_paths.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # colorbar stays so that it is easier to compare in plot

        # ax.view_init(30, Config.angle_for_net_plot)
        ax.view_init(90, Config.angle_for_net_plot)

        pdf.savefig(fig_paths)
        plt.close(fig_paths)

    if d == 1 or d == 2:
        def get_net(u, k):
            if not NN.single_net_algorithm():
                return u[k]
            else:
                return u[0]

        x = np.reshape(np.linspace(short[0], short[1], n_sample_points), (n_sample_points, 1)) * np.ones((1, d))

        # TODO: recall: it is vitaly important that plot_number=k+1+NN.k
        l = NN.N
        NN = ProminentResults.disc_best_result.load_state_dict_into_given_net(NN)
        for k in range(l):
            draw_function(x, get_net(NN.u, k), plot_number=k+1+NN.K, algorithm=NN.single_net_algorithm())
        NN = ProminentResults.cont_best_result.load_state_dict_into_given_net(NN)
        for k in range(l):
            draw_function(x, get_net(NN.u, k), plot_number=k+1+NN.K, algorithm=NN.single_net_algorithm())
        NN = ProminentResults.final_result.load_state_dict_into_given_net(NN)
        for k in range(l):
            draw_function(x, get_net(NN.u, k), plot_number=k+1+NN.K, algorithm=NN.single_net_algorithm())

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
