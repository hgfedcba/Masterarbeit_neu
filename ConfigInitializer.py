from ModelDefinitions import add_mu_c_x, add_sigma_c_x, add_american_put, add_bermudan_max_call, binomial_trees
from ModelDefinitions import mu_dict, payoff_dict, sigma_dict

from MathematicalModel import MathematicalModel

from NetDefinitions import add_am_call_default_pretrain, add_am_put_default_pretrain, add_multiplicative_lr_scheduler, pretrain_functions, lr_decay_algs, optimizers, add_step_lr_scheduler
from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid, id

from sklearn.model_selection import ParameterGrid

from Config import Config

import numpy as np
import matplotlib.pyplot as plt

import NN
import copy

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
        intro_string = None
        current_Config = None
        test_paths = None
        val_paths = None
        angle_for_net_plot = None

        result_list = []
        if option == 4312:
            # American put in 1d
            # This Model is stupid since it results in no sells whatsoever

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

            add_am_put_default_pretrain(K, 16)

            max_minutes = 60  # //5
            # not more, N=50!
            train_size = 256
            test_size = 512
            val_size = 4048
            """
            # works but takes too long
            batch_size = 1024
            test_size = 2048
            val_size = 16384
            """
            x_plot_range_for_net_plot = [10, 50]

            Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            Model.set_reference_value(5.318)  # verified with my binomial trees
            Model.update_parameter_string()

            test_paths_file = "../test_paths_4312.npy"
            val_paths_file = "../val_paths_4312.npy"
            test_paths = np.load(test_paths_file, mmap_mode="r")
            val_paths = np.load(val_paths_file, mmap_mode="r")

        elif option == 4411_2:
            # bermudan max call
            # TODO: graphikkarte (überraschend schwer)  andere pytorch installation!
            # TODO: path dependent option
            # TODO: Saved paths are not antithetic...
            # TODO: colorcode output
            # TODO: Robins-problem
            # TODO: Russian option

            # TODO: Plot average time of stopping for test set

            r = 0.05
            sigma_constant = 0.2  # beta
            mu_constant = r
            K = 100
            xi = 110
            T = 3
            N = 9
            d = 2  # dimension
            delta = 0.1  # dividend rate
            sigma = add_sigma_c_x(sigma_constant)
            mu = add_mu_c_x(mu_constant, delta)
            g = add_bermudan_max_call(K, r)

            add_am_call_default_pretrain(K+10, 60)

            max_minutes = 60  # TODO: change
            # batch_size = 8192
            # test_size = 8192
            # val_size = 16384

            train_size = 2048
            test_size = 4092
            val_size = 16384

            x_plot_range_for_net_plot = [60, 200]
            angle_for_net_plot = 225

            Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            Model.set_reference_value(21.344)
            Model.update_parameter_string()

            test_paths_file = "../test_paths_4411_2.npy"
            val_paths_file = "../val_paths_4411_2.npy"
            test_paths = np.load(test_paths_file, mmap_mode="r")
            val_paths = np.load(val_paths_file, mmap_mode="r")

        elif option == 4411_5:
            # bermudan max call

            r = 0.05
            sigma_constant = 0.2  # beta
            mu_constant = r
            K = 100
            xi = 110
            T = 3
            N = 9
            d = 5  # dimension
            delta = 0.1  # dividend rate
            sigma = add_sigma_c_x(sigma_constant)
            mu = add_mu_c_x(mu_constant, delta)
            g = add_bermudan_max_call(K, r)

            add_am_call_default_pretrain(K+10, 60)

            max_minutes = 60
            # batch_size = 8192
            # test_size = 8192
            # val_size = 16384

            train_size = 2048
            test_size = 4092
            val_size = 16384

            x_plot_range_for_net_plot = [60, 200]
            angle_for_net_plot = 225

            Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            Model.set_reference_value(36.763)
            Model.update_parameter_string()
            """
            test_paths = Model.generate_paths(1048576, True)
            val_paths = Model.generate_paths(1048576, True)

            np.save("../test_paths_4411_5.npy", test_paths)
            np.save("../val_paths_4411_5.npy", val_paths)
            """
            test_paths_file = "../test_paths_4411_5.npy"
            val_paths_file = "../val_paths_4411_5.npy"
            test_paths = np.load(test_paths_file, mmap_mode="r")
            val_paths = np.load(val_paths_file, mmap_mode="r")

        elif option == 2:
            # Model
            r = 0.05
            sigma_constant = 0.25  # beta
            mu_constant = r
            K = 40
            xi = 40
            T = 10
            N = 10
            d = 2  # dimension
            delta = 0  # dividend rate
            sigma = add_sigma_c_x(sigma_constant)
            mu = add_mu_c_x(mu_constant, delta)
            g = add_american_put(d, K, r)

            add_am_put_default_pretrain(K, 16)

            max_minutes = 5
            train_size = 64
            test_size = 256
            val_size = 2048

            x_plot_range_for_net_plot = [10, 50]
            angle_for_net_plot = 225  # TODO: remove after test is finished

            Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, 200, K))
            Model.update_parameter_string()

            test_paths_file = "../test_paths_2.npy"
            val_paths_file = "../val_paths_2.npy"
            test_paths = np.load(test_paths_file, mmap_mode="r")
            val_paths = np.load(val_paths_file, mmap_mode="r")

        elif option == 1:
            # Model
            r = 0.05
            sigma_constant = 0.25  # beta
            mu_constant = r
            K = 40
            xi = 40
            T = 10
            N = 10
            d = 1  # dimension
            delta = 0  # dividend rate
            sigma = add_sigma_c_x(sigma_constant)
            mu = add_mu_c_x(mu_constant, delta)
            g = add_american_put(d, K, r)

            add_am_put_default_pretrain(K, 16)

            max_minutes = 3
            train_size = 64
            test_size = 256
            val_size = 2048

            """
            # using 1 as an actual benchmark
            # Make new version with actual N
            max_minutes = 20
            batch_size = 512
            test_size = 1024
            val_size = 8192
            """
            x_plot_range_for_net_plot = [10, 50]

            Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            # Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, 2000, K))
            Model.set_reference_value(6.245555049146182)  # N = 2000
            Model.update_parameter_string()

            test_paths_file = "../test_paths_1.npy"
            val_paths_file = "../val_paths_1.npy"
            test_paths = np.load(test_paths_file, mmap_mode="r")
            val_paths = np.load(val_paths_file, mmap_mode="r")

        else:
            # Model
            r = 0.05
            sigma_constant = 0.25  # beta
            mu_constant = r
            K = 40
            xi = 40
            T = 10
            N = 10
            d = 1  # dimension
            delta = 0  # dividend rate
            sigma = add_sigma_c_x(sigma_constant)
            mu = add_mu_c_x(mu_constant, delta)
            g = add_american_put(d, K, r)

            add_am_put_default_pretrain(K, 16)

            max_minutes = 0.5*0.5
            train_size = 64
            test_size = 64
            val_size = 256

            x_plot_range_for_net_plot = [10, 50]

            Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, N*10, K))
            Model.update_parameter_string()

            test_paths_file = "../test_paths_1.npy"
            val_paths_file = "../val_paths_1.npy"
            test_paths = np.load(test_paths_file, mmap_mode="r")
            val_paths = np.load(val_paths_file, mmap_mode="r")

        # Parametergrid für Netz
        # addAdam
        add_step_lr_scheduler(500)
        add_multiplicative_lr_scheduler(0.999)

        list_individual_parameters = []
        list_common_parameters = []

        dict_a = {  #
            'algorithm'                : [2],
            'internal_neurons'         : [50, 100],  # 50?
            'hidden_layer_count'       : [3, 4],
            'activation_internal'      : [tanh],  # [tanh, relu, leaky_relu, softsign, selu]
            'activation_final'         : [sigmoid],
            'optimizer'                : [0],
            'pretrain_func'            : [False],  # 2 information in 1 entry "False" for pass
            'pretrain_iterations'      : [500],
            'max_number_of_iterations' : [10000],
            'max_minutes_of_iterations': [max_minutes],
            'initial_lr'               : [0.02],  # 0.01 for other setting
            'lr_decay_alg'             : [2],  # 2 Information in 1 entry
            'random_seed'              : [1337],
            'validation_frequency'     : [10],
            'antithetic_val'           : [True],  # ALWAYS TRUE, SINCE I LOAD FROM MEMORY
            'antithetic_train'         : [False],
            'train_size'               : [train_size],
            'test_size'                : [test_size],  # with my current implementation this has to be constant over a programm execution
            'val_size'                 : [val_size]  # with my current implementation this has to be constant over a programm execution
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
            optimizer = optimizers[params['optimizer']]
            if params['pretrain_func'] is False:
                do_pretrain = False
            else:
                do_pretrain = True
            pretrain_func = pretrain_functions[params['pretrain_func']]
            pretrain_iterations = params['pretrain_iterations']
            max_number_of_iterations = params['max_number_of_iterations']
            max_minutes_of_iterations = params['max_minutes_of_iterations']

            train_size = params['train_size']
            initial_lr = params['initial_lr']
            if params['lr_decay_alg'] is False:
                do_lr_decay = False
                lr_decay_alg = params['lr_decay_alg']
            else:
                do_lr_decay = True
                lr_decay_alg = lr_decay_algs[params['lr_decay_alg']]
            random_seed = params['random_seed']
            validation_frequency = params['validation_frequency']
            antithetic_val = params['antithetic_val']
            antithetic_train = params['antithetic_val']
            test_size = params['test_size']
            val_size = params['val_size']

            current_Config = Config(algorithm, internal_neurons, hidden_layer_count, activation_internal, activation_final, optimizer, do_pretrain, pretrain_func, pretrain_iterations,
                                    max_number_of_iterations,
                                    max_minutes_of_iterations, train_size, initial_lr, do_lr_decay, lr_decay_alg, random_seed, validation_frequency, antithetic_val, antithetic_train, test_size,
                                    val_size, stop_paths_in_plot, x_plot_range_for_net_plot, angle_for_net_plot)
            if run_number == 0:
                f = open("intermediate_results.txt", "w")
                intro_string = "Wir optimieren für das Modell: \t" + Model.parameter_string + "Folgende Parameter sind konstant über alle runs: \t" + \
                              current_Config.get_psl_wrt_list(list_common_parameters) + "\nLegende: a\t(b)\t | \tc\t(d)\t" + \
                              "Vor dem Strich stehen die diskreten Werte, hinter dem Strich die stetigen. In Klammern sind die Werte aus der final validation angegeben\n\n"
                f.write(intro_string)
                f.close()

                log.warning("The reference value is: " + str(Model.get_reference_value()))

                test_paths = test_paths[:test_size]
                val_paths = val_paths[:val_size]

            # Rufe main_routine auf und erhalte result
            individual_parameter_string = current_Config.get_psl_wrt_list(list_individual_parameters)
            individual_parameter_list = current_Config.get_pl_wrt_list(list_individual_parameters)

            log.warning("This is run " + str(run_number) + " and the current config is: " + individual_parameter_string + "\n")
            '''
            if algorithm == 3:
                Model_copied = copy.deepcopy(Model)
                Model.__N = 10
                current_NN = NN.NN(current_Config, Model_copied, Memory, log, test_paths)

                # result enthält prominent_result klasse, memory klasse
                optimitaion_result = [current_NN.optimization()]
            '''
            current_NN = NN.NN(current_Config, Model, Memory, log)
            m_out = 0

            if algorithm == 3:  # later 3
                N_factor = 2
                if N % N_factor != 0:
                    N_factor = 3
                    if N % N_factor != 0:
                        log.critical("Algorithm 3 doesn't work")
                # shorten test paths
                shortened_test_paths = test_paths[:, :, ::N_factor]

                Model.setN(Model.getN()//N_factor)
                current_NN.M_max = 50
                m_out = current_NN.optimization(shortened_test_paths, m_out)[0]
                current_NN.M_max = max_number_of_iterations
                Model.setN(Model.getN()*N_factor)
                log.warning("Alg 3 \"pretrain\" ends")

                # deletes old Prominent Results
                current_NN.ProminentResults.initialize_empty()

            optimitaion_result = [current_NN.optimization(test_paths, m_out)[1:]]
            log.warning("Final val begins")
            fvs = time.time()
            optimitaion_result[0][0].final_validation(val_paths)
            Memory.final_val_duration = time.time() - fvs
            Memory.end_time = time.time()

            result_list.append([optimitaion_result[0][0], optimitaion_result[0][1], run_number, individual_parameter_string, individual_parameter_list])

            log.warning("Plotting begins\n\n")
            f = open("intermediate_results.txt", "a")
            f.write(self.result_to_resultstring(result_list[-1]))
            f.close()

            Out.create_graphics(Memory, optimitaion_result[0][0], Model, current_Config, run_number, test_paths, val_paths, current_NN)

            run_number += 1

        def sort_resultlist_by_highest_disc_value_on_val_set(result_list):
            def sort_key(element):
                # return -max(element[0].disc_best_result.val_disc_value, element[0].cont_best_result.val_disc_value, element[0].final_result.val_disc_value)
                return -max(element[0].disc_best_result.val_disc_value, element[0].cont_best_result.val_disc_value, element[0].final_result.val_disc_value)
            result_list.sort(key=sort_key)

        sort_resultlist_by_highest_disc_value_on_val_set(result_list)

        f = open("end_result.txt", "w")
        f.write(intro_string)
        for res in result_list:
            f.write(self.result_to_resultstring(res))
        f.close()

        self.create_outputtable(Model, current_Config, list_common_parameters, result_list)

    @staticmethod
    def result_to_resultstring(result):
        def short_disc(a):
            return str(round(a.test_disc_value, 5)) + " \t (" + str(round(a.val_disc_value, 5)) + ")\t"

        def short_cont(a):
            return str(round(a.test_cont_value, 5)) + " \t (" + str(round(a.val_cont_value, 5)) + ")\t"

        os = mylog("\trun: ", str(result[2]),
                   "best discrete result:", short_disc(result[0].disc_best_result), " | ", short_cont(result[0].disc_best_result),
                   "\tbest cont result:", short_disc(result[0].cont_best_result), " | ", short_cont(result[0].cont_best_result),
                   "\tfinal result:", short_disc(result[0].final_result), " | ", short_cont(result[0].final_result),
                   "\ttime taken until discrete/cont/final result:", result[0].disc_best_result.time_to_this_result, " | ", result[0].cont_best_result.time_to_this_result, " | ",
                   result[1].end_time - result[1].start_time,
                   "\titerations taken until discrete/cont/final result:", result[0].disc_best_result.m, " | ", result[0].cont_best_result.m, " | ", result[0].final_result.m,
                   "\ttime spend training:", sum(result[1].train_durations), "time spend testing:", sum(result[1].val_durations), "time spend on net:", sum(result[1].total_net_durations),
                   "time spend on pretrain:", result[1].pretrain_duration, "time spend on final val:", result[1].final_val_duration,
                   "Parameterstring:", result[3], only_return=True)
        return os

    # discrete final/best disc auf final daten, time until for both    -  variable parameter
    @staticmethod
    def create_outputtable(Model, current_config, list_common_parameters, resultlist):
        # TODO: Colorcode name vs value
        title_text1 = "Wir optimieren für das Modell: \t" + Model.parameter_string + "\n\nFolgende Parameter sind konstant über alle Runs:\n"
        title_text2 = current_config.get_psl_wrt_list(list_common_parameters)

        for k in range(1, 4):
            index = title_text2.find("\t", 150*k)
            title_text2 = title_text2[:index] + '\n' + title_text2[index:]

        title_text = title_text1 + title_text2
        title_text = title_text.replace('\t', '    ')

        footer_text = 'stub'
        fig_background_color = 'skyblue'
        fig_border = 'steelblue'

        data = []
        # too long
        '''
        data.append(['average payoff of on the validation set using the net giving the best result on the test set', 'average payoff of on the validation set using the final net',
                     'time until intermediate result', 'time total'])
        '''
        data.append(['best disc', 'best_cont', 'final', 'iterations'] + [param[0] for param in resultlist[0][4]])
        for res in resultlist:
            data.append(['  ' + str(res[2]) + '  ', res[0].disc_best_result.val_disc_value, res[0].cont_best_result.val_disc_value, res[0].final_result.val_disc_value,
                         str(res[0].disc_best_result.m) + " | "+ str(res[0].cont_best_result.m) + " | " + str(res[0].final_result.m)]
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
                   figsize=(10, 3)
                   )  # Add a table at the bottom of the axes
        h = [0.1] * 5
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
