import Alg10, Alg20
import ModelInitializer
import Util

from NetDefinitions import add_am_call_default_pretrain, add_am_put_default_pretrain, add_multiplicative_lr_scheduler, pretrain_functions, lr_decay_algs, optimizers, add_step_lr_scheduler
# noinspection PyUnresolvedReferences
from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid, id

from sklearn.model_selection import ParameterGrid

from Config import Config

import numpy as np
import matplotlib.pyplot as plt

import NN

from Util import mylog
import time

from Memory import Memory as MemeClass

import Out

from RobbinsModel import RobbinsModel
from W_RobbinsModel import W_RobbinsModel


class ConfigInitializer:
    def __init__(self, option, log):
        # Here i first choose the option i want to price. For every kind of option i implement a parameter grid that contains all parameters that are used for the option. I then define a model
        # class that contains all stats of the theoretical model including prices i have from other sources. i also define a new instance of the config class for every element of the parameter grid.
        # i later use this concrete config for the nets etc.

        start_time = time.time()
        intro_string = None
        current_Config = None

        result_list = []
        val_paths, test_paths, angle_for_net_plot, max_number, max_minutes, train_size, val_size, test_size, Model, x_plot_range_for_net_plot, val_paths_file = ModelInitializer.initialize_model(option)

        # Parametergrid für Netz
        # addAdam
        add_step_lr_scheduler(500)
        add_multiplicative_lr_scheduler(0.999)
        add_multiplicative_lr_scheduler(0.994)  # this halves the learning rate compared to the one above at 150 iterations

        list_individual_parameters = []
        list_common_parameters = []
        # [Adam, Adadelta, Adagrad, AdamW, Adamax, ASGD, RMSprop, SGD]

        # assert not self.single_net_algorithm() or not isinstance(Model, RobbinsModel)
        dict_a = {  #
            'device'                                : ["cpu"],  # ["cpu", "cuda:0"]
            'algorithm'                             : [5],  # 5, 0, 21, 20, 15
            'sort net input'                        : [True],  # remember: val and test list are sorted, for alg 21 I load val_paths again
            'pretrain with empty nets'              : [True, False],  # TODO: think about how I handle the difference between alg 20 and alg 21
            'internal neurons per layer'            : [50],  # 50, 100
            'hidden layer count'                    : [2],  # [1, 2, 3]
            'internal activation function'          : [tanh],  # [tanh, relu, leaky_relu, softsign, selu]
            'final activation function'             : [sigmoid],
            'optimizer'                             : [72],  # [7, 72] [2, 7, 71, 72, 73] [0, 2, 3, 4, 7, 71, 72, 73] ... 1, 5, 8 scheinen schlechter, 7 besonders gut.
            # Wenn 2 -> _, dann 21 -> _ mit den ersten besonderen einstellungen.
            'pretrain function'                     : [False],  # 2 information in 1 entry "False" for pass
            'number pretrain iterations'            : [500],
            'max number of iterations'              : [max_number],
            'max minutes of iterations'             : [max_minutes],
            # [0.02] + 0.999 und [0.05] + 0.994 haben sich beide bewährt
            'initial lr'                            : [0.005],  # [0.005, 0.02] 0.01 for other setting
            'lr decay algorithm'                    : [3],  # [2, 3] 2 Information in 1 entry
            'dropout rate'                          : [0],  # only 0, breaks alg20
            'random seed'                           : [1337],
            'validation frequency'                  : [10],
            'antithetic variables on validation set': [True],  # ALWAYS TRUE, SINCE I LOAD FROM MEMORY
            'antithetic variables on train set'     : [False],
            'training batch size'                   : [train_size],
            'number of validation paths'            : [val_size],  # with my current implementation this has to be constant over a programm execution
            'number of test paths'                  : [test_size]  # with my current implementation this has to be constant over a programm execution
        }

        for u, v in dict_a.items():
            if v.__len__() > 1:
                if u == "pretrain function":
                    list_individual_parameters.append("pretrain")
                if u == "lr decay algorithm":
                    list_individual_parameters.append("do lr decay")
                list_individual_parameters.append(u)
            else:
                if u == "pretrain function":
                    list_common_parameters.append("pretrain")
                if u == "lr decay algorithm":
                    list_common_parameters.append("do lr decay")
                list_common_parameters.append(u)

        run_number = 0

        for params in ParameterGrid(dict_a):
            Memory = MemeClass()

            device = params['device']
            stop_paths_in_plot = False
            algorithm = params['algorithm']
            sort_net_input = params['sort net input']
            pretrain_with_empty_nets = params['pretrain with empty nets']
            if not (isinstance(Model, RobbinsModel) or isinstance(Model, W_RobbinsModel)):
                sort_net_input = False
            internal_neurons = params['internal neurons per layer']
            hidden_layer_count = params['hidden layer count']
            activation_internal = params['internal activation function']
            activation_final = params['final activation function']
            optimizer_number = params['optimizer']
            if params['pretrain function'] is False:
                do_pretrain = False
            else:
                do_pretrain = True
            pretrain_func = pretrain_functions[params['pretrain function']]
            if algorithm == 5:
                algorithm = 0  # TODO: this is wrong
                do_pretrain = True
                pretrain_func = 0
            pretrain_iterations = params['number pretrain iterations']
            max_number_of_iterations = params['max number of iterations']
            max_minutes_of_iterations = params['max minutes of iterations']

            train_size = params['training batch size']
            initial_lr = params['initial lr']
            if params['lr decay algorithm'] is False:
                do_lr_decay = False
                lr_decay_alg = params['lr decay algorithm']
            else:
                do_lr_decay = True
                lr_decay_alg = lr_decay_algs[params['lr decay algorithm']]
            dropout_rate = params['dropout rate']
            random_seed = params['random seed']
            validation_frequency = params['validation frequency']
            if algorithm >= 10:
                validation_frequency = 1
            antithetic_val = params['antithetic variables on validation set']
            antithetic_train = params['antithetic variables on train set']

            current_Config = Config(device, algorithm, sort_net_input, pretrain_with_empty_nets, internal_neurons, hidden_layer_count, activation_internal, activation_final, optimizer_number, do_pretrain, pretrain_func, pretrain_iterations,
                                    max_number_of_iterations,
                                    max_minutes_of_iterations, train_size, initial_lr, do_lr_decay, lr_decay_alg, dropout_rate, random_seed, validation_frequency, antithetic_val, antithetic_train, test_size,
                                    val_size, stop_paths_in_plot, x_plot_range_for_net_plot, angle_for_net_plot)
            if run_number == 0:
                f = open("intermediate_results.txt", "w")
                intro_string = "Wir optimieren \t" + Model.parameter_string + "Folgende Parameter sind konstant über alle runs: \t" + \
                               current_Config.get_psl_wrt_list(list_common_parameters) + "\nLegende: a\t(b)\t | \tc\t(d)\t" + \
                               "Vor dem Strich stehen die diskreten Werte, hinter dem Strich die stetigen. In Klammern sind die Werte aus dem Test angegeben\n\n"
                f.write(intro_string)
                f.close()

                log.warning("The reference value is: " + str(Model.get_reference_value()))

                val_paths = val_paths[:val_size]  # TODO: parameter, der hier die letzten pfade wählt
                test_paths = test_paths[:test_size]

                if sort_net_input:
                    Util.sort_list_inplace(val_paths)
                    Util.sort_list_inplace(test_paths)

            # Rufe main_routine auf und erhalte result
            individual_parameter_string = current_Config.get_psl_wrt_list(list_individual_parameters)
            individual_parameter_list = current_Config.get_pl_wrt_list(list_individual_parameters)

            log.warning("This is run " + str(run_number) + " and the current config is: " + individual_parameter_string + "\n")
            '''
            if algorithm == 3:
                Model_copied = copy.deepcopy(Model)
                Model.__N = 10
                current_NN = NN.NN(current_Config, Model_copied, Memory, log, testxxx_paths)

                # result enthält prominent_result klasse, memory klasse
                optimitaion_result = [current_NN.optimization()]
            '''
            if 20 > algorithm >= 10:
                current_NN = Alg10.Alg10_NN(current_Config, Model, Memory, log, val_paths_file=val_paths_file)
                if algorithm == 11 or algorithm >= 14:
                    current_NN.do_pretrain = True
            elif algorithm >= 20:
                current_NN = Alg20.Alg20_NN(current_Config, Model, Memory, log, val_paths_file=val_paths_file)
            else:
                current_NN = NN.NN(current_Config, Model, Memory, log, val_paths_file=val_paths_file)

            m_out = 0

            if algorithm == 3:  # later 3
                N_factor = 2
                if Model.getN() % N_factor != 0:
                    N_factor = 3
                    if Model.getN() % N_factor != 0:
                        log.critical("Algorithm 3 doesn't work")
                # shorten val paths
                shortened_val_paths = val_paths[:, :, ::N_factor]

                Model.setN(Model.getN() // N_factor)
                current_NN.M_max = 50
                m_out = current_NN.optimization(shortened_val_paths, m_out)[0]
                current_NN.M_max = max_number_of_iterations
                Model.setN(Model.getN() * N_factor)
                log.warning("Alg 3 \"pretrain\" ends")

                # deletes old Prominent Results
                current_NN.ProminentResults.initialize_empty()
            optimization_result = [current_NN.optimization(val_paths, m_out)[1:]]
            log.warning("Test begins")
            fvs = time.time()
            final = optimization_result[0][0].test(test_paths)
            log.info("Testing on the final net gives: " + str(final))
            Memory.test_duration = time.time() - fvs
            Memory.end_time = time.time()

            result_list.append([optimization_result[0][0], optimization_result[0][1], run_number, individual_parameter_string, individual_parameter_list])

            log.warning("Plotting begins\n\n")
            f = open("intermediate_results.txt", "a")
            f.write(self.result_to_resultstring(result_list[-1], algorithm))
            f.close()

            Out.create_graphics(Memory, optimization_result[0][0], Model, current_Config, run_number, val_paths, test_paths, current_NN)

            run_number += 1

        def sort_resultlist_by_highest_disc_value_on_test_set(result_list):
            def sort_key(element):
                return -max(element[0].disc_best_result.test_disc_value, element[0].cont_best_result.test_disc_value, element[0].final_result.test_disc_value)

            result_list.sort(key=sort_key)

        sort_resultlist_by_highest_disc_value_on_test_set(result_list)

        f = open("end_result.txt", "w")
        f.write(intro_string)
        for res in result_list:
            f.write(self.result_to_resultstring(res, algorithm))
        f.close()

        self.create_outputtable(Model, current_Config, list_common_parameters, result_list)

    @staticmethod
    def result_to_resultstring(result, alg):
        def short_disc(a):
            return Util.force_5_decimal(a.val_cont_value) + " \t (" + Util.force_5_decimal(a.test_disc_value) + ")\t"

        def short_cont(a):
            return Util.force_5_decimal(a.val_cont_value) + " \t (" + Util.force_5_decimal(a.test_cont_value) + ")\t"

        """
        def short_disc(a):
            return str(round(a.val_disc_value, 5)) + " \t (" + str(round(a.test_disc_value, 5)) + ")\t"

        def short_cont(a):
            return str(round(a.val_cont_value, 5)) + " \t (" + str(round(a.test_cont_value, 5)) + ")\t"
        """

        # TODO: use ljust more
        if alg >= 10:
            os = mylog("\trun: ", str(result[2]),
                       "\tamount of times without stopping:", result[0].final_result.amount_of_times_where_no_stopping_happens,
                       "best discrete result:", short_disc(result[0].disc_best_result), " | ", short_cont(result[0].disc_best_result),
                       "\tbest cont result:", short_disc(result[0].cont_best_result), " | ", short_cont(result[0].cont_best_result),
                       "\tfinal result:", short_disc(result[0].final_result), " | ", short_cont(result[0].final_result),
                       "\ttime taken until discrete/cont/final result:", result[0].disc_best_result.time_to_this_result, " | ", result[0].cont_best_result.time_to_this_result, " | ",
                       result[1].end_time - result[1].start_time,
                       "\titerations taken until discrete/cont/final result:", str(len(result[1].average_train_payoffs)).ljust(30, " "),
                       "\ttime spend training:", sum(result[1].single_train_durations), "time spend testing:", sum(result[1].val_durations), "time spend on net:", sum(result[1].total_net_durations_per_validation),
                       "time spend on pretrain:", result[1].pretrain_duration, "time spend on final val:", result[1].test_duration,
                       "Parameterstring:", result[3])
        else:
            os = mylog("\trun: ", str(result[2]),
                       "\tamount of times without stopping:", result[0].final_result.amount_of_times_where_no_stopping_happens,
                       "best discrete result:", short_disc(result[0].disc_best_result), " | ", short_cont(result[0].disc_best_result),
                       "\tbest cont result:", short_disc(result[0].cont_best_result), " | ", short_cont(result[0].cont_best_result),
                       "\tfinal result:", short_disc(result[0].final_result), " | ", short_cont(result[0].final_result),
                       "\ttime taken until discrete/cont/final result:", result[0].disc_best_result.time_to_this_result, " | ", result[0].cont_best_result.time_to_this_result, " | ",
                       result[1].end_time - result[1].start_time,
                       "\titerations taken until discrete/cont/final result:", result[0].disc_best_result.m, " | ", result[0].cont_best_result.m, " | ", result[0].final_result.m,
                       "\ttime spend training:", sum(result[1].single_train_durations), "time spend testing:", sum(result[1].val_durations), "time spend on net:", sum(result[1].total_net_durations_per_validation),
                       "time spend on pretrain:", result[1].pretrain_duration, "time spend on final val:", result[1].test_duration,
                       "Parameterstring:", result[3])

        return os

    # discrete final/best disc auf final daten, time until for both    -  variable parameter
    @staticmethod
    def create_outputtable(Model, current_config, list_common_parameters, resultlist):
        # Colorcode name vs value?
        title_text1 = "Wir optimieren für das Modell: \n" + Model.parameter_string + "\n\nFolgende Parameter sind konstant über alle Runs:\n"
        title_text2 = current_config.get_psl_wrt_list(list_common_parameters)

        for k in range(1, 4):
            index = title_text2.find("\t", 150 * k)
            title_text2 = title_text2[:index] + '\n' + title_text2[index:]

        title_text = title_text1 + title_text2
        title_text = title_text.replace('\t', '    ')

        # footer_text = 'stub'
        fig_background_color = 'skyblue'
        fig_border = 'steelblue'

        data = []
        # too long
        '''
        data.append(['average payoff of on the test set using the net giving the best result on the val set', 'average payoff of on the test set using the final net',
                     'time until intermediate result', 'time total'])
        '''
        data.append(['best disc', 'best cont', 'final', 'iterations', 'time'] + [param[0] for param in resultlist[0][4]])
        for res in resultlist:
            if not (isinstance(res[0].NN, Alg10.Alg10_NN) or isinstance(res[0].NN, Alg20.Alg20_NN)):
                data.append(['  ' + str(res[2]) + '  ', res[0].disc_best_result.test_disc_value, res[0].cont_best_result.test_disc_value, res[0].final_result.test_disc_value, str(res[0].disc_best_result.m) + " | " + str(res[0].cont_best_result.m) + " | " + str(res[0].final_result.m), res[1].end_time-res[1].start_time] + [str(param[1]) for param in res[4]])
            else:
                data.append(['  ' + str(res[2]) + '  ', res[0].disc_best_result.test_disc_value, res[0].cont_best_result.test_disc_value, res[0].final_result.test_disc_value,
                             str(len(res[1].average_train_payoffs)), res[1].end_time - res[1].start_time]
                            + [str(param[1]) for param in res[4]])

        for i in range(1, data.__len__()):
            for j in range(data[1].__len__()):
                if isinstance(data[i][j], float):
                    data[i][j] = round(data[i][j], 3)
        # Pop the headers from the data array
        column_headers = data.pop(0)
        row_headers = [x.pop(0) for x in data]  # Table data needs to be non-numeric text. Format the data
        # while I'm at it.
        cell_text = []
        for row in data:
            cell_text.append([str(x) for x in row])  # Get some lists of color specs for row and column headers
        rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
        # Create the figure. Setting a small pad on tight_layout seems to better regulate white space. Sometimes experimenting with an explicit figsize here can produce better outcome.
        plt.figure(linewidth=2,
                   edgecolor=fig_border,
                   facecolor=fig_background_color,
                   tight_layout={'pad': 1},
                   figsize=(10, 3 + data.__len__() / 5)  # figsize=(10, 3)   gerade 2.5 / 4
                   )  # Add a table at the bottom of the axes
        # my current guess 20.10.21: Es wird die breite der zellen angegeben von links nach recht und wenn h zu lang ist wir es zurechtgeschnitten. Warum die erste Spalte klein ist weiß ich nicht.
        h = [0.07, 0.07, 0.07, 0.1, 0.07]
        h.extend([0.15] * 10)
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
        # plt.figtext(0.95, 0.05, horizontalalignment='right', size=6, weight='light')  # Force the figure to update, so backends center objects correctly within the figure.
        # Without plt.draw() here, the title will center on the axes and not the figure.
        plt.draw()  # Create image. plt.savefig ignores figure edge and face colors, so map them.
        fig = plt.gcf()
        plt.savefig('Overview.png',
                    # bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=500  # AUFLÖSUNG
                    )
