from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid, pretrain_functions, lr_decay_algs, pretrain_func_dict, activation_func_dict, lr_decay_dict, optimizer_dict


class Config:
    # Ich gebe Standardwerte an um den Datentyp zu deklarieren. Ich möchte die Standardwerte in fast allen Fällen überschreiben.
    def __init__(self, device="cpu", algorithm=0, sort_net_input=False, internal_neurons=50, hidden_layer_count=3, activation_internal=tanh, activation_final=sigmoid, optimizer_number=Adam, do_pretrain=True,
                 pretrain_func=pretrain_functions[0], pretrain_iterations=800,
                 max_number_of_iterations=50, max_minutes_of_iterations=5, train_size=32, initial_lr=0.0001, do_lr_decay=False, lr_decay_alg=lr_decay_algs[0], dropout_rate=0, random_seed=23343,
                 validation_frequency=2, antithetic_val=True, antithetic_train=False, val_size=64, test_size=128, stop_paths_in_plot=False, x_plot_range_for_net_plot=None, angle_for_net_plot=40):

        # net

        # Algorithmus 0 macht genau was ich möchte. u_n bezeichnet die Wahrscheinlichkeit das es gut ist im n-ten Schritt zu stoppen. U_n bezeichnet die Wahrscheinlichkeit das im n_ten Schritt
        # gestoppt wird. Es wird diskret gestoppt wenn u_n > 0.5 ist.

        if x_plot_range_for_net_plot is None:
            x_plot_range_for_net_plot = [20, 60]
        self.device = device
        self.algorithm = algorithm  # 0 is source, 2 is christensen learn f, 3 is different N
        self.sort_net_input = sort_net_input
        self.internal_neurons = internal_neurons
        self.hidden_layer_count = hidden_layer_count
        self.activation_internal = activation_internal
        self.activation_final = activation_final
        self.optimizer_number = optimizer_number

        self.do_pretrain = do_pretrain
        self.pretrain_func = pretrain_func
        self.pretrain_iterations = pretrain_iterations

        self.max_number_iterations = max_number_of_iterations
        self.max_minutes_of_iteration = max_minutes_of_iterations
        self.train_size = train_size
        self.initial_lr = initial_lr  # lernrate
        self.do_lr_decay = do_lr_decay
        self.lr_decay_alg = lr_decay_alg
        self.dropout_rate = dropout_rate

        # Meta
        self.random_seed = random_seed
        self.validation_frequency = validation_frequency
        self.antithetic_val = antithetic_val  # only in validation!
        self.antithetic_train = antithetic_train
        self.val_size = val_size
        self.test_size = test_size

        self.stop_paths_in_plot = stop_paths_in_plot

        self.x_plot_range_for_net_plot = x_plot_range_for_net_plot
        self.angle_for_net_plot = angle_for_net_plot

        alg_dict = {
            # TODO: I hardcoded that algs 10-19 are similar to 10 and 20-29 are similar to 20, algs >= 10 learn sequentially
            0: "Paper",
            2: "single Net",
            3: "smaller N pretrain",
            5: "Paper with empty pretrain",
            10: "back to front",
            11: "seq with pretrain",
            12: "explicit stop condition given",  # doesn't run through
            14: "(14) back to front empty pretrain",
            15: "(15) train together at the end",
            16: "(16) train all nets after pretrain",
            20: "front to back N=const",
            21: "front to back N=inc"
        }

        pl = [["device", device], ["algorithm", alg_dict.get(algorithm)], ["sort net input", sort_net_input], ["internal neurons per layer", internal_neurons], ["hidden layer count", hidden_layer_count],
              ["internal activation function", activation_func_dict.get(activation_internal)], ["final activation function", activation_func_dict.get(activation_final)],
              ["optimizer", optimizer_dict.get(optimizer_number)], ["do pretrain", do_pretrain], ["pretrain function", pretrain_func_dict.get(pretrain_func)],
              ["number pretrain iterations", pretrain_iterations], ["max number of iterations", max_number_of_iterations],
              ["max minutes of iterations", max_minutes_of_iterations], ["initial lr", initial_lr], ["do lr decay", do_lr_decay], ["lr decay algorithm", lr_decay_dict.get(lr_decay_alg)],
              ['dropout rate', dropout_rate], ["random seed", random_seed], ["validation frequency", validation_frequency], ["antithetic variables on validation set", antithetic_val],
              ["antithetic variables on train set", antithetic_train], ["training batch size", train_size], ["number of validation paths", val_size], ["number of test paths", test_size],
              ["stop paths in plot", stop_paths_in_plot]]

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
