from Util import *
from ProminentResults import ProminentResults
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, d, internal_neurons, hidden_layer_count, activation_internal, activation_final):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(d, internal_neurons)
        self.fcs = []
        for _ in range(hidden_layer_count):
            self.fcs.append(nn.Linear(internal_neurons, internal_neurons))
        self.fcl = nn.Linear(internal_neurons, 1)

        self.activation_internal = activation_internal
        self.activation_final = activation_final
        self.hidden_layer_count = hidden_layer_count

    def forward(self, y):
        y = self.activation_internal(self.fc1(y))
        for k in range(self.hidden_layer_count):
            y = self.activation_internal(self.fcs[k](y))
        y = self.activation_final(self.fc3(y))
        return y


class NN:
    def __init__(self, Config, Model, Memory, log, val_paths):
        self.Memory = Memory
        self.ProminentResults = ProminentResults(log)
        self.log = log

        self.val_paths = val_paths

        self.Model = Model
        self.T = Model.getT()
        self.N = Model.getN()
        self.d = Model.getd()

        np.random.seed(Config.random_seed)
        torch.manual_seed(Config.random_seed)

        self.initial_lr = Config.initial_lr  # Lernrate
        self.lr_decay_alg = Config.lr_decay_alg
        self.do_lr_decay = Config.do_lr_decay
        self.final_val_size = Config.final_val_size

        self.internal_neurons = Config.internal_neurons
        self.activation_internal = Config.activation_internal
        self.activation_final = Config.activation_final
        self.hidden_layer_count = Config.hidden_layer_count
        self.optimizer = Config.optimizer

        self.do_pretrain = Config.do_pretrain
        self.pretrain_iterations = Config.pretrain_iterations
        self.pretrain_func = Config.pretrain_func

        self.validation_frequency = Config.validation_frequency
        self.antithetic_variables = Config.antithetic_variables

        self.algorithm = Config.algorithm

        self.t = Model.get_time_partition(self.N)
        self.u = []

        # TODO: use this
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def define_nets():
            self.u = []
            for _ in range(self.N):
                net = Net(self.d, self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final)
                self.u.append(net)

        define_nets()

    def optimization(self):
        log = self.log

        self.ProminentResults.process_current_iteration(None, 42, 1.7, 500, None, 5)
        self.ProminentResults.process_current_iteration(None, 40, 1.9, 300, None, 6)
        self.ProminentResults.set_final_result(None, 44, 1.7, 500, None, 5)
        self.Memory.train_durations.append(5)
        self.Memory.val_durations.append(17)
        self.Memory.total_net_durations.append(23)
        self.Memory.val_continuous_value_list.append(3)
        self.Memory.val_continuous_value_list.append(4)
        self.Memory.val_discrete_value_list.append(2.5)
        self.Memory.val_discrete_value_list.append(3.5)

        return self.ProminentResults, self.Memory

    def pretrain(self):
        assert True

    def train(self):
        assert True

    def validate(self):
        assert True
