import Alg10
import Alg20
import ModelInitializer
import Util

# noinspection PyUnresolvedReferences
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
from Shortened_RobbinsModel import Shortened_RobbinsModel
from Filled_RobbinsModel import Filled_RobbinsModel


# Dies ist die Run Klasse. Sie führt eine Serie von Tests aus und ist dafür zuständig alle anderen Klassen zu initialisieren und aufzurufen. Alles geschieht im Konstruktor.
class Run:
    def __init__(self, option, log):

        intro_string = None
        current_Config = None

        result_list = []

        # Initilisiere Modell
        val_paths, angle_for_net_plot, max_number, max_minutes, train_size, val_size, test_size, Model, x_plot_range_for_net_plot, val_paths_file, test_paths_file, last_paths =\
            ModelInitializer.initialize_model(option)

        # Parametergrid für Netz
        add_step_lr_scheduler(500)
        add_multiplicative_lr_scheduler(0.998)  # changed from 0.999 to 0.998 on 21.10.21
        add_multiplicative_lr_scheduler(0.994)  # this halves the learning rate compared to the one above at 150 iterations

        list_individual_parameters = []  # Eine Liste der Parameter, die sich in den Durchläufen ändern
        list_common_parameters = []  # Eine Liste der Parameter, die in allen Durchläufen identisch sind

        # Hier werden die verschiedenen Durchläufe eingestellt. Im Kommentar ist eventuell erklärt welche Konfigurationen es gibt/welche ich verwende
        dict_a = {  #
            'device'                                : ["cpu"],  # ["cpu", "cuda:0"]
            'algorithm'                             : [0],
            'sort net input'                        : [True],
            'pretrain with empty nets'              : [True],
            'internal neurons per layer'            : [50],  # 50, 100
            'hidden layer count'                    : [2],  # [2, 3]
            'internal activation function'          : [selu],  # [tanh, relu, leaky_relu, softsign, selu]
            'final activation function'             : [sigmoid],
            'optimizer'                             : [72],  # [2, 72]  ... 1, 5, 8 scheinen schlechter, 7 besonders gut.
            'pretrain function'                     : [False],  # 2 information in 1 entry "False" for pass
            'number pretrain iterations'            : [500],
            'max number of iterations'              : [max_number],
            'max minutes of iterations'             : [max_minutes],
            # Lernraten von [0.02] + 0.999 und [0.05] + 0.994 haben sich beide bewährt
            'initial lr'                            : [0.005],  # [0.005, 0.02]  Recall: Ich habe ein continue eingebaut damit nur die beiden guten konfigs genommen werden
            'lr decay algorithm'                    : [3],  # [2, 3] 2 Information in 1 entry
            'dropout rate'                          : [0],
            'random seed'                           : [1337],
            'validation frequency'                  : [10],
            'antithetic variables on validation set': [True],  # ALWAYS TRUE, SINCE I LOAD FROM MEMORY
            'antithetic variables on train set'     : [False],
            'training size during pretrain'         : [0.25],
            'training batch size'                   : [train_size],
            'number of validation paths'            : [val_size],
            'number of test paths'                  : [test_size]
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

        # Hier parse ich die Werte aus dem Parametergrid
        for params in ParameterGrid(dict_a):
            Memory = MemeClass()

            device = params['device']
            stop_paths_in_plot = False
            algorithm = params['algorithm']
            sort_net_input = params['sort net input']
            if isinstance(Model, Shortened_RobbinsModel) or isinstance(Model, Filled_RobbinsModel):
                sort_net_input = True
            pretrain_with_empty_nets = params['pretrain with empty nets']
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
                do_pretrain = True
                pretrain_func = None
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
            training_size_during_pretrain = params['training size during pretrain']

            current_Config = Config(device, algorithm, sort_net_input, pretrain_with_empty_nets, internal_neurons, hidden_layer_count, activation_internal, activation_final, optimizer_number,
                                    do_pretrain, pretrain_func, pretrain_iterations, max_number_of_iterations, max_minutes_of_iterations, training_size_during_pretrain, train_size, initial_lr,
                                    do_lr_decay, lr_decay_alg, dropout_rate, random_seed, validation_frequency, antithetic_val, antithetic_train, test_size, val_size, stop_paths_in_plot,
                                    x_plot_range_for_net_plot, angle_for_net_plot)
            if run_number == 0:
                f = open("intermediate_results.txt", "w")
                intro_string = "Wir optimieren \t" + Model.parameter_string + "Folgende Parameter sind konstant über alle runs: \t" + \
                               current_Config.get_psl_wrt_list(list_common_parameters) + "\nLegende: a\t(b)\t | \tc\t(d)\t" + \
                               "Vor dem Strich stehen die diskreten Werte, hinter dem Strich die stetigen. In Klammern sind die Werte aus dem Test angegeben\n\n"
                f.write(intro_string)
                f.close()

                log.warning("The reference value is: " + str(Model.get_reference_value()))

                if last_paths:
                    val_paths = val_paths[-val_size:]
                else:
                    val_paths = val_paths[:val_size]

            # Rufe main_routine auf und erhalte result
            individual_parameter_string = current_Config.get_psl_wrt_list(list_individual_parameters)
            individual_parameter_list = current_Config.get_pl_wrt_list(list_individual_parameters)

            log.warning("This is run " + str(run_number) + " and the current config is: " + individual_parameter_string + "\n")

            if (algorithm == 2 or algorithm == 3) and isinstance(Model, RobbinsModel):
                log.info("continued")
                continue

            if algorithm == 21 and not isinstance(Model, RobbinsModel):
                log.info("continued")
                continue
            if (lr_decay_alg == 3 and initial_lr == 0.05) or (lr_decay_alg == 2 and initial_lr == 0.02):  # added 17.12.21 after run 30
                continue

            if 20 > algorithm >= 10:
                current_NN = Alg10.Alg10_NN(current_Config, Model, Memory, log, val_paths_file=val_paths_file)
                if algorithm == 11 or algorithm >= 14:
                    current_NN.do_pretrain = True
            elif algorithm >= 20:
                current_NN = Alg20.Alg20_NN(current_Config, Model, Memory, log, val_paths_file=val_paths_file)
            else:
                # current_NN = NNKatagoInspired.KatagoNet(current_Config, Model, Memory, log, val_paths_file=val_paths_file)  # This would start the Katago-NN if I had come around to finish it
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

            # Hier findet das Training statt
            optimization_result = [current_NN.optimization(val_paths, m_out)[1:]]
            log.warning("Test begins")
            fvs = time.time()
            # Hier findet das Testen statt
            # Idee vom 8.3.22: Lade die pfade erst in der test funktion und nur weniger auf einmal
            test_paths, test_size = ModelInitializer.load_test_paths(test_paths_file, Model, test_size, last_paths)
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
            test_paths = None  # Wieso wird das nicht genutzt, die Schleife läuft doch weiter...

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

        assert len(result_list) > 0
        self.create_outputtable(Model, current_Config, list_common_parameters, result_list)

    @staticmethod
    def result_to_resultstring(result, alg):
        def short_disc(a):
            return Util.force_5_decimal(a.val_cont_value) + " \t (" + Util.force_5_decimal(a.test_disc_value) + ")\t"

        def short_cont(a):
            return Util.force_5_decimal(a.val_cont_value) + " \t (" + Util.force_5_decimal(a.test_cont_value) + ")\t"

        if alg >= 10:
            os = mylog("\trun: ", str(result[2]),
                       "\tamount of times without stopping:", result[0].final_result.amount_of_times_where_no_stopping_happens,
                       "best discrete result:", short_disc(result[0].disc_best_result), " | ", short_cont(result[0].disc_best_result),
                       "\tbest cont result:", short_disc(result[0].cont_best_result), " | ", short_cont(result[0].cont_best_result),
                       "\tfinal result:", short_disc(result[0].final_result), " | ", short_cont(result[0].final_result),
                       "\ttime taken until discrete/cont/final result:", result[0].disc_best_result.time_to_this_result, " | ", result[0].cont_best_result.time_to_this_result, " | ",
                       result[1].end_time - result[1].start_time,
                       "\titerations taken until final result:        ", str(len(result[1].average_train_payoffs)).ljust(30, " "),
                       "\ttime spend training:", sum(result[1].single_train_durations), "time spend testing:", sum(result[1].val_durations), "time spend on net:",
                       sum(result[1].total_net_durations_per_validation), "time spend on pretrain:", result[1].pretrain_duration, "time spend on final val:", result[1].test_duration,
                       "Parameterstring:", result[3], (result[1].net_resets != "") * "\tnet resets:", result[1].net_resets)
        else:
            os = mylog("\trun: ", str(result[2]),
                       "\tamount of times without stopping:", result[0].final_result.amount_of_times_where_no_stopping_happens,
                       "best discrete result:", short_disc(result[0].disc_best_result), " | ", short_cont(result[0].disc_best_result),
                       "\tbest cont result:", short_disc(result[0].cont_best_result), " | ", short_cont(result[0].cont_best_result),
                       "\tfinal result:", short_disc(result[0].final_result), " | ", short_cont(result[0].final_result),
                       "\ttime taken until discrete/cont/final result:", result[0].disc_best_result.time_to_this_result, " | ", result[0].cont_best_result.time_to_this_result, " | ",
                       result[1].end_time - result[1].start_time,
                       "\titerations taken until discrete/cont/final result:", result[0].disc_best_result.m, " | ", result[0].cont_best_result.m, " | ", result[0].final_result.m,
                       "\ttime spend training:", sum(result[1].single_train_durations), "time spend testing:", sum(result[1].val_durations), "time spend on net:",
                       sum(result[1].total_net_durations_per_validation), "time spend on pretrain:", result[1].pretrain_duration, "time spend on final val:", result[1].test_duration,
                       "Parameterstring:", result[3], (result[1].net_resets != "") * "\tnet resets:", result[1].net_resets)

        return os

    @staticmethod
    def create_outputtable(Model, current_config, list_common_parameters, resultlist):
        title_text1 = "Wir optimieren für das Modell: \n" + Model.parameter_string + "\n\nFolgende Parameter sind konstant über alle Runs:\n"
        title_text2 = current_config.get_psl_wrt_list(list_common_parameters)

        for k in range(1, 4):
            index = title_text2.find("\t", 150 * k)
            title_text2 = title_text2[:index] + '\n' + title_text2[index:]

        title_text = title_text1 + title_text2
        title_text = title_text.replace('\t', '    ')

        fig_background_color = 'skyblue'
        fig_border = 'steelblue'

        data = [['disc value', '# no stop', 'iterations', 'time'] + [param[0] for param in resultlist[0][4]]]
        for res in resultlist:
            # changed from highest disc to last disc after run 30
            data.append(['  ' + str(res[2]) + '  ', res[0].final_result.test_disc_value, res[0].final_result.amount_of_times_where_no_stopping_happens, str(len(res[1].average_train_payoffs)),
                         res[1].end_time - res[1].start_time] + [str(param[1]) for param in res[4]])

        for i in range(1, data.__len__()):
            for j in range(data[1].__len__()):
                if isinstance(data[i][j], float):
                    data[i][j] = round(data[i][j], 3)
        column_headers = data.pop(0)
        row_headers = [x.pop(0) for x in data]
        cell_text = []
        for row in data:
            cell_text.append([str(x) for x in row])
        rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
        ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
        plt.figure(linewidth=2,
                   edgecolor=fig_border,
                   facecolor=fig_background_color,
                   tight_layout={'pad': 1},
                   figsize=(10, 3 + data.__len__() / 5)  # figsize=(10, 3)   gerade 2.5 / 4
                   )
        # my current guess 20.10.21: Es wird die breite der zellen angegeben von links nach recht und wenn h zu lang ist wir es zurechtgeschnitten. Warum die erste Spalte klein ist weiß ich nicht.
        h = [0.07, 0.07, 0.1, 0.07]
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
        the_table.scale(1, 1)  # Hide axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  # Hide axes border
        plt.box(on=None)  # Add title
        plt.suptitle(title_text, fontsize=6)  # Add footer
        plt.draw()
        fig = plt.gcf()
        plt.savefig('Overview.png',
                    # bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=500  # Auflösung
                    )
