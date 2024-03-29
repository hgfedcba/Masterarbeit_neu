# noinspection PyUnresolvedReferences
from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid, pretrain_functions, lr_decay_algs, pretrain_func_dict, activation_func_dict, lr_decay_dict, optimizer_dict


# Hier speichere ich die Variablen, die einen Durchlauf beschreiben. Außerdem habe ich hier das alg_dict, in dem steht welcher Algorithmus welche Funktionalität hat.
class Config:
    # Ich gebe Standardwerte an um den Datentyp zu deklarieren. Ich möchte die Standardwerte in fast allen Fällen überschreiben.
    def __init__(self, device="cpu", algorithm=0, sort_net_input=False, pretrain_with_empty_nets=False, internal_neurons=50, hidden_layer_count=3, activation_internal=tanh, activation_final=sigmoid,
                 optimizer_number=Adam, do_pretrain=True, pretrain_func=None, pretrain_iterations=800, max_number_of_iterations=50, max_minutes_of_iterations=5, training_size_during_pretrain=1,
                 train_size=32, initial_lr=0.0001, do_lr_decay=False, lr_decay_alg=lr_decay_algs[0], dropout_rate=0, random_seed=23343, validation_frequency=2, antithetic_val=True,
                 antithetic_train=False, val_size=64, test_size=128, stop_paths_in_plot=False, x_plot_range_for_net_plot=None, angle_for_net_plot=40):

        # net

        # Algorithmus 0 macht genau was ich möchte. u_n bezeichnet die Wahrscheinlichkeit das es gut ist im n-ten Schritt zu stoppen. U_n bezeichnet die Wahrscheinlichkeit das im n_ten Schritt
        # gestoppt wird. Es wird diskret gestoppt wenn u_n > 0.5 ist.

        if x_plot_range_for_net_plot is None:
            x_plot_range_for_net_plot = [20, 60]
        self.device = device  # CPU oder GPU
        self.algorithm = algorithm  # see below
        self.sort_net_input = sort_net_input
        self.pretrain_with_empty_nets = pretrain_with_empty_nets  # Der Unterschied, ob nach dem Pretrain-Zeitpunkt weitere leere Netze und ZV sind
        self.internal_neurons = internal_neurons
        self.hidden_layer_count = hidden_layer_count
        self.activation_internal = activation_internal
        self.activation_final = activation_final
        self.optimizer_number = optimizer_number  # see below

        self.do_pretrain = do_pretrain
        self.pretrain_func = pretrain_func  # target function, only for finance settings
        self.pretrain_iterations = pretrain_iterations

        self.max_number_iterations = max_number_of_iterations
        self.max_minutes_of_iteration = max_minutes_of_iterations
        self.training_size_during_pretrain = training_size_during_pretrain
        self.train_size = train_size
        self.initial_lr = initial_lr  # lernrate
        self.do_lr_decay = do_lr_decay
        self.lr_decay_alg = lr_decay_alg
        self.dropout_rate = dropout_rate

        # Meta
        self.random_seed = random_seed
        self.validation_frequency = validation_frequency
        self.antithetic_val = antithetic_val  # antithetic variables in validation
        self.antithetic_train = antithetic_train  # antithetic variables in test
        self.val_size = val_size
        self.test_size = test_size

        self.stop_paths_in_plot = stop_paths_in_plot

        self.x_plot_range_for_net_plot = x_plot_range_for_net_plot
        self.angle_for_net_plot = angle_for_net_plot

        alg_dict = {
            # I hardcoded that algs 10-19 are similar to 10 (back to front) and 20-29 are similar to 20 (front to back), algs >= 10 learn sequentially
            0: "Paper",  # Kernalgorithmus
            2: "single Net",  # Ich habe nur ein Netz für alle Zeitpunkte
            3: "smaller N pretrain",  # Eine Variante von 2, bei der ich mich am Finite-Elemente-Verfahren aus der Vorlesung Wissenschaftliches Rechnen orientiere. In Kürze ist die Idee erst
            # einmal ein kleineres N zu betrachten und dann ein größeres
            5: "Paper with empty pretrain",  # Kernalgorithmus, bei dem jedes Netz vortrainiert wird
            6: "Paper with net resets",  # Kernalgorithmus, bei dem Netze zurückgesetzt werden, die nicht stoppen
            7: "Paper with both",  # Funktionalität von 5 und 6
            # folgende Algorithemn befinden sich in der ALG10 Klasse
            10: "back to front",  # Ich lerne die Netze von hinten nach vorne, erste implementation
            11: "seq with pretrain",  # abgebrochen
            12: "explicit stop condition given",  # abgebrochen
            14: "(14) back to front empty pretrain",  # Die folgenden Netze sind leere Netze, da sonst kein Training stattfindet
            15: "(15) train together at the end",  # Es wird am Ende zusammen trainiert
            16: "(16) train all nets after pretrain",  # Diese Version beschreibe ich in der Arbeit
            # folgende Algorithmen befinden sich in der ALg20 Klasse
            20: "front to back N=const",  # Die Varainte 2 aus der Arbeit
            21: "front to back N=inc"  # Die Variante 1 aus der Arbeit
        }

        pl = [["device", device], ["algorithm", alg_dict.get(algorithm)], ["sort net input", sort_net_input], ["pretrain with empty nets", pretrain_with_empty_nets],
              ["internal neurons per layer", internal_neurons], ["hidden layer count", hidden_layer_count], ["internal activation function", activation_func_dict.get(activation_internal)],
              ["final activation function", activation_func_dict.get(activation_final)], ["optimizer", optimizer_dict.get(optimizer_number)], ["do pretrain", do_pretrain],
              ["pretrain function", pretrain_func_dict.get(pretrain_func)], ["number pretrain iterations", pretrain_iterations], ["max number of iterations", max_number_of_iterations],
              ["max minutes of iterations", max_minutes_of_iterations], ["initial lr", initial_lr], ["do lr decay", do_lr_decay], ["lr decay algorithm", lr_decay_dict.get(lr_decay_alg)],
              ['dropout rate', dropout_rate], ["random seed", random_seed], ["validation frequency", validation_frequency], ["antithetic variables on validation set", antithetic_val],
              ["antithetic variables on train set", antithetic_train], ['training size during pretrain', training_size_during_pretrain], ["training batch size", train_size],
              ["number of validation paths", val_size], ["number of test paths", test_size], ["stop paths in plot", stop_paths_in_plot]]

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
