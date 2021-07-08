from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid, pretrain_functions, lr_decay_algs, pretrain_func_dict, activation_func_dict, lr_decay_dict, optimizer_dict


class Config:
    # Ich gebe Standardwerte an um den Datentyp zu deklarieren. Ich möchte die Standardwerte in fast allen Fällen überschreiben.
    def __init__(self, algorithm=0, internal_neurons=50, hidden_layer_count=3, activation_internal=tanh, activation_final=sigmoid, optimizer=Adam, do_pretrain=True,
                 pretrain_func=pretrain_functions[0], pretrain_iterations=800,
                 max_number_of_iterations=50, max_minutes_of_iterations=5, train_size=32, initial_lr=0.0001, do_lr_decay=False, lr_decay_alg=lr_decay_algs[0], random_seed=23343,
                 validation_frequency=2, antithetic_val=True, antithetic_train=False, test_size=64, val_size=128, stop_paths_in_plot=False, x_plot_range_for_net_plot=None, angle_for_net_plot=40):

        # net

        # Algorithmus 0 macht genau was ich möchte. u_n bezeichnet die Wahrscheinlichkeit das es gut ist im n-ten Schritt zu stoppen. U_n bezeichnet die Wahrscheinlichkeit das im n_ten Schritt
        # gestoppt wird. Es wird diskret gestoppt wenn u_n > 0.5 ist.

        if x_plot_range_for_net_plot is None:
            x_plot_range_for_net_plot = [20, 60]
        self.algorithm = algorithm  # 0 is source, 2 is christensen learn f, 3 is different N
        self.internal_neurons = internal_neurons
        self.hidden_layer_count = hidden_layer_count
        self.activation_internal = activation_internal
        self.activation_final = activation_final
        self.optimizer = optimizer

        self.do_pretrain = do_pretrain
        self.pretrain_func = pretrain_func
        self.pretrain_iterations = pretrain_iterations

        self.max_number_iterations = max_number_of_iterations
        self.max_minutes_of_iteration = max_minutes_of_iterations
        self.train_size = train_size
        self.initial_lr = initial_lr  # lernrate
        self.do_lr_decay = do_lr_decay
        self.lr_decay_alg = lr_decay_alg

        # Meta
        self.random_seed = random_seed
        self.validation_frequency = validation_frequency
        self.antithetic_val = antithetic_val  # only in validation!
        self.antithetic_train = antithetic_train
        self.test_size = test_size
        self.val_size = val_size

        self.stop_paths_in_plot = stop_paths_in_plot

        self.x_plot_range_for_net_plot = x_plot_range_for_net_plot
        self.angle_for_net_plot = angle_for_net_plot

        alg_dict = {
            0: "Paper",
            2: "single Net",
            3: "smaller N pretrain",
            10: "train sequentially",
            11: "seq with pretrain"
        }

        # TODO: entferne Unterstriche, verbessere Beschreibung:  Trivialer Ansatz geht nicht!
        pl = [["algorithm", alg_dict.get(algorithm)], ["internal_neurons", internal_neurons], ["hidden_layer_count", hidden_layer_count], ["activation_internal", activation_func_dict.get(activation_internal)],
              ["activation_final", activation_func_dict.get(activation_final)], ["optimizer", optimizer_dict.get(optimizer)], ["do_pretrain", do_pretrain],
              ["pretrain_func", pretrain_func_dict.get(pretrain_func)], ["pretrain_iterations", pretrain_iterations], ["max_number_of_iterations", max_number_of_iterations],
              ["max_minutes_of_iterations", max_minutes_of_iterations], ["initial_lr", initial_lr], ["do_lr_decay", do_lr_decay], ["lr_decay_alg", lr_decay_dict.get(lr_decay_alg)],
              ["random_seed", random_seed], ["validation_frequency", validation_frequency], ["antithetic_val", antithetic_val], ["antithetic_train", antithetic_train], ["train_size", train_size],
              ["test_size", test_size], ["val_size", val_size], ["stop_paths_in_plot", stop_paths_in_plot]]

        self.parameter_list = pl

    def get_psl_wrt_list(self, l):  # get parameter string list with resect to list
        out = ""
        for s in self.get_pl_wrt_list(l):
            out += s[0] + ": " + str(s[1]) + " \t"
        return out

    def get_pl_wrt_list(self, l):  # get parameter list with resect to list
        out = []
        for s in self.parameter_list:
            if l.__contains__(s[0]):
                out.append(s)
        return out

    """
    parameter_string = "algorithm: ", algorithm, "internal_neurons: ", internal_neurons, "activation1: ", activation_func_dict.get(activation1), "activation2: ", \
                       activation_func_dict.get(activation2), "optimizer: ", optimizer_dict.get(optimizer), "pretrain: ", pretrain, "pretrain_func: ", pretrain_func_dict.get(pretrain_func), \
                       "pretrain_iterations: ", pretrain_iterations, "max_number_of_iterations: ", max_number_of_iterations, "max_minutes_of_iterations: ", max_minutes_of_iterations, \
                       "batch_size: ", batch_size, "initial_lr: ", initial_lr, "do_lr_decay: ", do_lr_decay, "lr_decay_alg: ", lr_decay_dict.get(lr_decay_alg), "random_seed: ", random_seed, \
                       "validation_frequency: ", validation_frequency, "antithetic_variables: ", antithetic_variables, "val_size: ", val_size, "final_val_size: ", final_val_size, \
                       "stop_paths_in_plot: ", stop_paths_in_plot

    parameter_string = ''.join(str(s) + " \t" for s in parameter_string)
    self.parameter_string = parameter_string + "\n"
    """
    # Das Modell kann hier eigentlich raus, das brauche ich nur in ConfigInitializer
    """
    # Model
    self.T
    self.N
    self.xi
    self.d = 1  # dimension
    self.r = 0.05  # interest rate
    self.K = 40  # strike price
    self.delta = 0  # dividend rate
    self.sigma_constant = 0.25
    self.mu_constant = self.r
    self.sigma = self.sigma_c_x
    self.mu = self.mu_c_x
    self.g = self.american_put
    """
