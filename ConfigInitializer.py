from ModelDefinitions import add_mu_c_x, add_sigma_c_x, add_american_put, add_bermudan_max_call, binomial_trees
from ModelDefinitions import mu_dict, payoff_dict, sigma_dict

from MathematicalModel import MathematicalModel

from NetDefinitions import add_am_call_default_pretrain, add_am_put_default_pretrain, add_multiplicative_lr_sheduler, pretrain_functions, lr_decay_algs, optimizers
from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid

from sklearn.model_selection import ParameterGrid

from Config import Config

import numpy as np
import matplotlib.pyplot as plt

import NN

from Util import mylog
import time

from Memory import Memory as MemeClass

import Out

class ConfigInitializer:
    def __init__(self, option, log):
        # Here i first choose the option i want to price. For every kind of option i implement a parameter grid that contains all parameters that are used for the option. I then define a model
        # class that contains all stats of the theoretical model including prices i have from other sources. i also define a new instance of the config class for every element of the parameter grid.
        # i later use this concrete config for the nets etc.

        start_time = time.time()
        introstring = None
        current_Config = None

        result_list = []

        if option == 0:
            assert True
        elif option == 4312:
            # American put in 1d

            # Model
            r = 0.06
            sigma_constant = 0.4  # beta
            mu_constant = r
            K = 40
            xi = 40
            T = 1
            N = 50
            d = 1  # dimension
            delta = 0  # dividend rate
            sigma = add_sigma_c_x(sigma_constant)
            mu = add_mu_c_x(mu_constant, delta)
            g = add_american_put(d, K, r)

            Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, N, K))
            Model.update_parameter_string()

            # Parametergrid für Netz
            # addAdam
            add_am_put_default_pretrain(K, 14)  # TODO:14!
            # add_lr

            list_individual_parameters = []
            list_common_parameters = []

            dict_a = {  #
                'algorithm'                : [0],
                'internal_neurons'         : [50, 500],
                'hidden_layer_count'              : [3],
                'activation_internal'              : [tanh],
                'activation_final'              : [sigmoid],
                'optimizer'                : [0],
                'pretrain_func'            : [1, False],  # 2 information in 1 entry "False" for pass
                'pretrain_iterations'      : [800],
                'max_number_of_iterations' : [50],
                'max_minutes_of_iterations': [5],
                'batch_size'               : [32],
                'initial_lr'               : [0.0001],
                'lr_decay_alg'             : [False],  # 2 Information in 1 entry
                'random_seed'              : [23343],
                'validation_frequency'     : [2],
                'antithetic_variables'     : [True],
                'val_size'                 : [64],
                'final_val_size'           : [128]
            }

            for u, v in dict_a.items():
                if v.__len__() > 1:
                    if u == "pretrain_func":
                        list_individual_parameters.append("pretrain")
                    if u == "lr_decay_alg":
                        list_individual_parameters.append("do_lr_decay")
                    list_individual_parameters.append(u)
                else:
                    if u == "pretrain_func":
                        list_common_parameters.append("pretrain")
                    if u == "lr_decay_alg":
                        list_common_parameters.append("do_lr_decay")
                    list_common_parameters.append(u)

            run_number = 0

            # ->visual indicator<-
            for params in ParameterGrid(dict_a):
                Memory = MemeClass()

                stop_paths_in_plot = False
                algorithm = params['algorithm']
                internal_neurons = params['internal_neurons']
                hidden_layer_count = params['hidden_layer_count']
                activation_internal = params['activation_internal']
                activation_final = params['activation_final']
                optimizer = params['optimizer']
                if params['pretrain_func'] is False:
                    do_pretrain = False
                else:
                    do_pretrain = True
                pretrain_func = pretrain_functions[params['pretrain_func']]
                pretrain_iterations = params['pretrain_iterations']
                max_number_of_iterations = params['max_number_of_iterations']
                max_minutes_of_iterations = params['max_minutes_of_iterations']
                batch_size = params['batch_size']
                initial_lr = params['initial_lr']
                if params['lr_decay_alg'] is False:
                    do_lr_decay = False
                else:
                    do_lr_decay = True
                lr_decay_alg = params['lr_decay_alg']
                random_seed = params['random_seed']
                validation_frequency = params['validation_frequency']
                antithetic_variables = params['antithetic_variables']
                val_size = params['val_size']
                final_val_size = params['final_val_size']

                current_Config = Config(algorithm, internal_neurons, hidden_layer_count, activation_internal, activation_final, optimizer, do_pretrain, pretrain_func, pretrain_iterations, max_number_of_iterations,
                                        max_minutes_of_iterations, batch_size, initial_lr, do_lr_decay, lr_decay_alg, random_seed, validation_frequency, antithetic_variables, val_size,
                                        final_val_size, stop_paths_in_plot)
                if run_number == 0:
                    f = open("intermediate_results.txt", "w")
                    introstring = "Wir optimieren für das Modell: \t" + Model.parameter_string + "Folgende Parameter sind konstant über alle runs: \t" + \
                                  current_Config.get_psl_wrt_list(list_common_parameters) + "\nLegende: a\t(b)\t | \tc\t(d)\t" + \
                                  "Vor dem Strich stehen die diskreten Werte, hinter dem Strich die stetigen. In Klammern sind die Werte aus der final validation angegeben\n\n"
                    f.write(introstring)
                    f.close()

                # TODO: Here something important should happen
                # Rufe main_routine auf und erhalte result
                individual_parameter_string = current_Config.get_psl_wrt_list(list_individual_parameters)

                current_NN = NN.NN(current_Config, Model, Memory, log)

                # result enthält prominent_result klasse, durations klasse
                optimitaion_result = [current_NN.optimization()]
                optimitaion_result[0][0].final_validation(42)
                # TODO: final val möchte ich eventuell außerhalb von optimization aufrufen. Es hat den Vorteil das es einfacher ist die Wege für final val zu verwalten
                result_list.append([optimitaion_result[0][0], optimitaion_result[0][1], run_number, individual_parameter_string])

                f = open("intermediate_result.txt", "a")
                f.write(self.result_to_resultstring(result_list[-1]))
                f.close()

                Out.create_graphics(Memory, optimitaion_result[0][0], Model, current_Config, run_number)

                run_number += 1

            def sort_resultlist_by_disc_value(result_list):
                def sort_key(element):
                    return -element[0].disc_best_result.disc_value
                result_list.sort(key=sort_key)

            sort_resultlist_by_disc_value(result_list)

            f = open("end_result.txt", "w")
            f.write(introstring)
            for res in result_list:
                f.write(self.result_to_resultstring(res))
            f.close()

            # TODO: actually implement table

            self.create_outputtable(Model, current_Config, list_common_parameters, list_individual_parameters, result_list)

    @staticmethod
    def result_to_resultstring(result):
        def short_disc(a):
            return str(a.disc_value) + " \t (" + str(a.final_disc_value) + ")\t"

        def short_cont(a):
            return str(a.cont_value) + " \t (" + str(a.final_cont_value) + ")\t"

        os = mylog("\trun: ", str(result[2]),
                   "best discrete result:", short_disc(result[0].disc_best_result), " | ", short_cont(result[0].disc_best_result),
                   "\tbest cont result:", short_disc(result[0].cont_best_result), " | ", short_cont(result[0].cont_best_result),
                   "\tfinal result:", short_disc(result[0].final_result), " | ", short_cont(result[0].final_result),
                   "\trespective times:", result[0].disc_best_result.time_to_this_result, " | ", result[0].cont_best_result.time_to_this_result, " | ", time.time()-result[1].start_time,
                   "\ttime spend training:", sum(result[1].train_durations), "time spend validating:", sum(result[1].val_durations), "time spend on net:", sum(result[1].total_net_durations),
                   "Parameterstring:", result[3], only_return=True)
        return os

    @staticmethod
    def create_outputtable(Model, current_config, list_common_parameters, list_individual_parameters, resultlist):
        title_text = "Wir optimieren für das Modell: " + Model.parameter_string + "Folgende Parameter sind konstant über alle Runs: " + current_config.get_psl_wrt_list(list_common_parameters)
        title_text = title_text.replace('\t', '  ')
        footer_text = 'stub'
        fig_background_color = 'skyblue'
        fig_border = 'steelblue'

        individual_parameter_list = current_config.get_pl_wrt_list(list_individual_parameters)

        data = [
            ['Freeze', 'W', 'Flood', 'Quake', 'Hail', 'max_number_of_iterations', 'max_minutes_of_iterations', 'validation_frequency'],
            [66386, 174296, 75131, 577908, 32015, 32015, 32015, 32015],
            [58230, 381139, 78045, 99308, 160454, 32015, 32015, 32015],
            [89135, 80552, 152558, 'Ens', 603535, 32015, 32015, 32015],
            [78415, 81858, 150656, 193263, 69638, 32015, 32015, 32015],
            [139361, 331509, 343164, 781380, 52269, 32015, 32015, 32015],
        ]  # Pop the headers from the data array
        column_headers = data.pop(0)
        row_headers = [x.pop(0) for x in data]  # Table data needs to be non-numeric text. Format the data
        # while I'm at it.
        cell_text = []
        for row in data:
            cell_text.append([str(x) for x in row])  # Get some lists of color specs for row and column headers
        rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))  # Create the figure. Setting a small pad on tight_layout
        # seems to better regulate white space. Sometimes experimenting
        # with an explicit figsize here can produce better outcome.
        plt.figure(linewidth=2,
                   edgecolor=fig_border,
                   facecolor=fig_background_color,
                   tight_layout={'pad': 1},
                   figsize=(10, 3)
                   )  # Add a table at the bottom of the axes
        h = [0.05] * 5
        h.extend([0.15] * 5)
        the_table = plt.table(cellText=cell_text,
                              colWidths=h,
                              rowLabels=row_headers,
                              rowColours=rcolors,
                              rowLoc='right',
                              colColours=ccolors,
                              colLabels=column_headers,
                              loc='center',
                              cellLoc='center')  # Scaling is the only influence we have over top and bottom cell padding.
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(6)
        # Make the rows taller (i.e., make cell y scale larger).
        the_table.scale(1, 1)  # Hide axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  # Hide axes border
        plt.box(on=None)  # Add title
        plt.suptitle(title_text, fontsize=6)  # Add footer
        plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=6, weight='light')  # Force the figure to update, so backends center objects correctly within the figure.
        # Without plt.draw() here, the title will center on the axes and not the figure.
        plt.draw()  # Create image. plt.savefig ignores figure edge and face colors, so map them.
        fig = plt.gcf()
        plt.savefig('pyplot-table-demo.png',
                    # bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=500  # AUFLÖSUNG
                    )


