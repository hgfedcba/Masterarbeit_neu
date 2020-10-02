from ModelDefinitions import add_mu_c_x, add_sigma_c_x, add_american_put, add_bermudan_max_call, binomial_trees
from ModelDefinitions import mu_dict, payoff_dict, sigma_dict

from MathematicalModel import MathematicalModel

from NetDefinitions import add_am_call_default_pretrain, add_am_put_default_pretrain, add_multiplicative_lr_sheduler, pretrain_functions, lr_decay_algs, optimizers
from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid

from sklearn.model_selection import ParameterGrid

from Config import Config


class ConfigInitializer:
    def __init__(self, option):
        # Here i first choose the option i want to price. For every kind of option i implement a parameter grid that contains all parameters that are used for the option. I then define a model
        # class that contains all stats of the theoretical model including prices i have from other sources. i also define a new instance of the config class for every element of the parameter grid.
        # i later use this concrete config for the nets etc.

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

            self.Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            self.Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, N, K))
            self.Model.update_parameter_string()

            # Parametergrid für Netz
            # addAdam
            add_am_put_default_pretrain(K, 14)  # TODO:14!
            # add_lr

            for params in ParameterGrid({  #
                'algorithm'                : [0],
                'internal_neurons'         : [50],
                'activation1'              : [tanh],
                'activation2'              : [sigmoid],
                'optimizer'                : [0],
                'pretrain_func'            : [1],  # 2 information in 1 entry "False" for pass
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
            }):
                stop_paths_in_plot = False
                algorithm = params['algorithm']
                internal_neurons = params['internal_neurons']
                activation1 = params['activation1']
                activation2 = params['activation2']
                optimizer = params['optimizer']
                if params['pretrain_func'] is False:
                    pretrain = False
                else:
                    pretrain = True
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

                self.current_config = Config(algorithm, internal_neurons, activation1, activation2, optimizer, pretrain, pretrain_func, pretrain_iterations, max_number_of_iterations,
                                             max_minutes_of_iterations, batch_size, initial_lr, do_lr_decay, lr_decay_alg, random_seed, validation_frequency, antithetic_variables, val_size,
                                             final_val_size, stop_paths_in_plot)

                self.current_parameterstring = self.Model.parameter_string + self.current_config.parameter_string
                print(self.current_parameterstring)
