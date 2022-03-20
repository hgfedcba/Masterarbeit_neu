# Dieser Algorithmus ist inspiriert durch Katago
# Besonderheiten:
# - Wir übergeben dem Netz den erwarteten Rang wenn wir jetzt stoppen
# - Jedes Netz hat eine weitere Ausgabe und zwar eine vorhersage des Ranges, den wir am Ende erreichen
import numpy as np

import W_RobbinsModel
from Util import *
from ProminentResults import ProminentResults
import torch.nn as nn
import pytest
from RobbinsModel import RobbinsModel
from NetDefinitions import optimizers
from NN import NN

class KatagoNet(NN):
    def __init__(self, Config, Model, Memory, log, val_paths_file=None):
        self.Memory = Memory
        self.log = log

        self.Model = Model
        self.T = Model.getT()
        self.N = Model.getN()
        self.path_dim = Model.getpath_dim()
        self.t = Model.t
        self.K = Model.getK()

        np.random.seed(Config.random_seed)
        torch.manual_seed(Config.random_seed)

        self.T_max = Config.max_minutes_of_iteration
        self.M_max = Config.max_number_iterations

        self.initial_lr = Config.initial_lr  # Lernrate
        self.lr_decay_alg = Config.lr_decay_alg
        self.do_lr_decay = Config.do_lr_decay
        self.dropout_rate = Config.dropout_rate

        self.training_size_during_pretrain = Config.training_size_during_pretrain
        self.batch_size = Config.train_size

        self.internal_neurons = Config.internal_neurons
        self.activation_internal = Config.activation_internal
        self.activation_final = Config.activation_final
        self.hidden_layer_count = Config.hidden_layer_count
        self.optimizer_number = Config.optimizer_number

        self.do_pretrain = Config.do_pretrain
        self.pretrain_iterations = Config.pretrain_iterations
        self.pretrain_func = Config.pretrain_func
        self.pretrain_range = Config.x_plot_range_for_net_plot

        self.validation_frequency = Config.validation_frequency
        self.antithetic_train = Config.antithetic_train

        self.device = Config.device
        self.algorithm = Config.algorithm
        self.sort_net_input = Config.sort_net_input
        self.pretrain_with_empty_nets = Config.pretrain_with_empty_nets
        self.u = []

        self.val_paths_file = val_paths_file

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def define_nets():
            self.u = []
            if self.single_net_algorithm():
                assert np.allclose(self.path_dim, np.ones_like(self.path_dim) * self.path_dim[0])
                # net = Net(self.path_dim[0] + 1, self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, Model.getK(), self.device, self.dropout_rate)
                # net.to(self.device)
                net = self.define_net_with_path_dim_k(self.path_dim[0] + 2, out_dim=2)
                self.u.append(net)
            else:
                for k in range(self.N):
                    # net = Net(self.path_dim[k], self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, Model.getK(), self.device, self.dropout_rate)
                    # net.to(self.device)
                    net = self.define_net_with_path_dim_k(self.path_dim[k] + 1, out_dim=2)
                    self.u.append(net)

        assert not self.single_net_algorithm() or not isinstance(self.Model, RobbinsModel)

        define_nets()
        self.ProminentResults = ProminentResults(log, self)

    def optimization(self, val_paths, m_out):
        super().optimization(val_paths, m_out)

    def return_expected_ranks_of_paths(self, paths):
        N = self.N
        expected_ranks = []
        for j in range(len(paths)):
            expected_ranks.append([])
            for k in range(len(paths[j])):
                e = (N-k)*paths[j][k][-1]
                t = np.zeros(k+1)
                t[k] = 1
                h = np.asarray(paths[j][k])
                y = np.argsort(h)
                z1 = np.ones_like(y)
                z1[y] = np.arange(1, h.size + 1)
                e += np.dot(t, z1)
                expected_ranks[j].append(e)

        return expected_ranks

    def training_step(self, optimizer, training_paths=None, net_list=None):
        if training_paths is None:
            training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train)
            if self.sort_net_input:
                if isinstance(self.Model, RobbinsModel):
                    sort_lists_inplace_except_last_one(training_paths)
                else:
                    training_paths = sort_np_inplace(training_paths)
            U = torch.empty(len(training_paths), self.N + 1, device=self.device)

        else:
            if isinstance(training_paths[0], list):
                local_N = training_paths[0].__len__()
            else:
                local_N = training_paths[0].shape[1]
            U = torch.empty(len(training_paths), local_N, device=self.device)

        expected_ranks = self.return_expected_ranks_of_paths(training_paths)

        # breaks alg20
        for net in self.u:
            if net != fake_net:
                net.train(mode=True)

        individual_payoffs = []

        for j in range(len(training_paths)):
            h, _ = self.generate_stopping_time_factors_and_discrete_stoppoint_from_path(training_paths[j], True, net_list=net_list, expected_rank_path=expected_ranks[j])
            U[j, :] = h[:, 0]
            individual_payoffs.append(self.Model.calculate_payoffs(U[j, :], training_paths[j], self.Model.getg, self.t, device=self.device))
        average_payoff = torch.sum(torch.stack(individual_payoffs)) / len(individual_payoffs)

        loss = -average_payoff

        t = time.time()
        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
        self.Memory.total_net_durations_per_validation[-1] += time.time() - t

        return average_payoff.item()

    def validate(self, paths, net_list=None):
        if net_list is not None:
            N = len(net_list)  # This is only for alg 21
        else:
            N = self.N

        L = len(paths)
        if self.sort_net_input:
            if isinstance(self.Model, RobbinsModel):
                paths = sort_lists_inplace_except_last_one(paths, N=N)
            else:
                paths = sort_np_inplace(paths, in_place=False)

        cont_individual_payoffs = []
        disc_individual_payoffs = []
        stopping_times = []

        # breaks alg20
        for net in self.u:
            if net != fake_net:
                net.eval()

        # h = []
        U = torch.empty(L, N + 1, device=self.device)
        tau_list = []
        for l in range(L):
            mylog(l)
            pre_u, tau = self.generate_stopping_time_factors_and_discrete_stoppoint_from_path(paths[l][:N+1], False, net_list=net_list)
            U[l, :] = pre_u[:, 0]
            cont_individual_payoffs.append(self.Model.calculate_payoffs(U[l, :], paths[l][:N+1], self.Model.getg, self.t, device=self.device))

            # part 2: discrete
            tau_list.append(tau)

            single_stopping_time = np.zeros(N + 1)
            single_stopping_time[tau_list[l]] = 1
            disc_individual_payoffs.append(self.Model.calculate_payoffs(single_stopping_time, paths[l][:N+1], self.Model.getg, self.t).item())
            stopping_times.append(single_stopping_time)

        disc_payoff = sum(disc_individual_payoffs) / L
        temp = torch.sum(torch.stack(cont_individual_payoffs)) / L
        cont_payoff = temp.item()

        # tau list is better then stopping_times
        return cont_payoff, disc_payoff, stopping_times

    def generate_discrete_stopping_time_from_u(self, u):
        # between 0 and N
        for n in range(self.N + 1):
            if u[n] > 0.5:
                return n
        return self.N

    def generate_stopping_time_factors_and_discrete_stoppoint_from_path(self, x_input, grad, net_list=None, expected_rank_path=None):
        if net_list is None:
            net_list = self.u
        if isinstance(x_input, list):
            local_N = x_input.__len__()
        else:
            local_N = x_input.shape[1]
        assert len(net_list)+1 == local_N or self.single_net_algorithm(), "First is " + str(len(net_list)+1) + " and second is " + str(local_N)

        U = []
        probability_sum = []
        x = []
        # x = torch.from_numpy(x_input) doesn't work for some reason
        import itertools
        local_u = []
        for n in range(local_N):
            if n > 0:
                probability_sum.append(probability_sum[n - 1] + U[n - 1])  # 0...n-1
            else:
                probability_sum.append(torch.zeros(1, device=self.device))
            if n < local_N-1:
                t = time.time()
                if not self.single_net_algorithm():
                    if isinstance(x_input, list):
                        x.append(torch.tensor(itertools.chain(expected_rank_path[n], x_input[n][:]), dtype=torch.float32, requires_grad=grad, device=self.device))
                    else:
                        x.append(torch.tensor(x_input[:, n], dtype=torch.float32, requires_grad=grad, device=self.device))
                    local_u.append(net_list[n](x[n]))
                else:
                    into = np.append(n+self.K, x_input[:, n])  # Der Input ist der Zeitpunkt und der tatsächliche Aktienwert. Ich addiere self.K auf den Zeitpunkt da dieser Faktor später noch
                    # abgezogen wird und ich möglichst nahe an der 0 bleiben möchte.
                    x.append(torch.tensor(into, dtype=torch.float32, requires_grad=grad, device=self.device))
                    local_u.append(net_list[0](x[n]))
                self.Memory.total_net_durations_per_validation[-1] += time.time() - t
            else:
                local_u.append(torch.ones(1, device=self.device))  # TODO: ich stecke hier
            if isinstance(local_u[n], int) or isinstance(local_u[n], float):
                U.append(local_u[n] * (torch.ones(1) - probability_sum[n]))
            else:
                U.append(local_u[n] * (torch.ones(1, device=self.device) - probability_sum[n]))

        z = torch.stack(U)
        assert torch.sum(z).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(z).item())
        return z, self.generate_discrete_stopping_time_from_u(local_u)

