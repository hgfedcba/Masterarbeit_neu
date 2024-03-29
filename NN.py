import W_RobbinsModel
from Util import *
from ProminentResults import ProminentResults
import torch.nn as nn
import pytest
from RobbinsModel import RobbinsModel
from NetDefinitions import optimizers


# Hier nutze ich das torch Modul um das Netz zu definieren
# TODO: IMPORTANT NOTE: I subtract K from the input of the Network to ensure the learning works from the beginning on
class Net(nn.Module):
    def __init__(self, d, internal_neurons, hidden_layer_count, activation_internal, activation_final, K, device, dropout_rate=0, out=1):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(d, internal_neurons).to(device)
        self.fcs = []
        for _ in range(hidden_layer_count+1):
            self.fcs.append(nn.Linear(internal_neurons, internal_neurons).to(device))
            # self.fcs.append(nn.Linear(internal_neurons, internal_neurons).cuda())
        self.fcl = nn.Linear(internal_neurons, out).to(device)

        self.activation_internal = activation_internal
        self.activation_final = activation_final
        self.hidden_layer_count = hidden_layer_count+1

        self.K = K

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, y):
        y = y-self.K
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.activation_internal(self.fc1(y))
        for k in range(self.hidden_layer_count):
            y = self.activation_internal(self.fcs[k](y))
        y = self.activation_final(self.fcl(y))
        return y


# Diese Klasse ist dafür zuständig das Netz zu definieren, trainieren und auszuwerten
class NN:
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
                net = self.define_net_with_path_dim_k(self.path_dim[0] + 1)
                self.u.append(net)
            else:
                for k in range(self.N):
                    # net = Net(self.path_dim[k], self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, Model.getK(), self.device, self.dropout_rate)
                    # net.to(self.device)
                    net = self.define_net_with_path_dim_k(self.path_dim[k])
                    self.u.append(net)

        assert not self.single_net_algorithm() or not isinstance(self.Model, RobbinsModel)

        define_nets()
        self.ProminentResults = ProminentResults(log, self)

    def define_net_with_path_dim_k(self, k, out_dim=1):
        net = Net(k, self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, self.Model.getK(), self.device, self.dropout_rate, out=out_dim)
        net.to(self.device)
        return net

    def return_net_a_at_value_b(self, a, b):
        b = b.to(self.device)
        return self.u[a](b)

    # This checks if the algorithm consists of a single net or multiple nets.
    def single_net_algorithm(self):
        if self.algorithm == 2 or self.algorithm == 3:
            return True
        if self.algorithm == 0 or self.algorithm == 5 or self.algorithm == 6 or self.algorithm == 7 or self.algorithm >= 10:
            return False
        assert False

    # defines the optimizer from optimizer_number and perhaps additional parameters
    def return_optimizer(self, parameters, lr=None):
        if lr is None:
            lr = self.initial_lr
        if self.optimizer_number < 10:
            return optimizers[self.optimizer_number](parameters, lr=lr)
        elif self.optimizer_number == 71:
            return optimizers[self.optimizer_number // 10](parameters, lr=lr, centered=True)
        elif self.optimizer_number == 72:
            return optimizers[self.optimizer_number // 10](parameters, lr=lr, momentum=0.5)
        elif self.optimizer_number == 73:
            return optimizers[self.optimizer_number // 10](parameters, lr=lr, momentum=0.9)
        elif self.optimizer_number == 31:
            return optimizers[self.optimizer_number // 10](parameters, lr=lr, amsgrad=True)
        elif self.optimizer_number == 81:  # Unused
            return optimizers[self.optimizer_number // 10](parameters, lr=lr, nesterov=True)
        elif self.optimizer_number == 82:
            return optimizers[self.optimizer_number // 10](parameters, lr=lr, nesterov=True, momentum=0.5)
        elif self.optimizer_number == 83:
            return optimizers[self.optimizer_number // 10](parameters, lr=lr, momentum=0.5, dampening=0.5)

    # This function controls the training procedure.
    def optimization(self, val_paths, m_out):
        self.N = self.Model.getN()

        log = self.log
        # scheduler = None

        params = []
        for k in range(len(self.u)):
            params += list(self.u[k].parameters())
        optimizer = self.return_optimizer(params)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates

        # do pretrain if it is supposed to
        pretrain_start = time.time()
        if self.do_pretrain:
            log.info("pretrain starts")
            self.Memory.average_pretrain_payoffs = self.pretrain(self.T_max/2, min(self.M_max/2, 60 * self.N))
            self.Memory.pretrain_net_duration = self.Memory.total_net_durations_per_validation.pop()
        self.Memory.pretrain_duration = time.time() - pretrain_start
        if self.Memory.pretrain_duration > 0.1:
            log.info("pretrain took \t%s seconds" % self.Memory.pretrain_duration)

        self.Memory.train_durations_per_validation.append(0)
        self.Memory.total_net_durations_per_validation.append(0)

        # starts the training loop
        m = m_out
        # continues if:
        # 1. In der letzten Iteration keine validation stattgefunden hat
        # 2. noch kaum iteriert wurde
        # Eines der folgenden gilt:
        # 3.a die maximale zeit nicht überschritten wurde
        # 3.b es keine maximale Iterationszahl gibt oder sie noch nicht erreicht wurde
        # 3.c es in den letzten 200 Iterationen ein neues optimum gab
        while (m % self.validation_frequency != 1 and not self.validation_frequency == 1) or \
                ((time.time() - self.Memory.start_time) / 60 < self.T_max and (self.M_max == -1 or m < self.M_max) and self.ProminentResults.get_m_max() + 200 > m)\
                or m < 30:
            m_th_iteration_start_time = time.time()

            average_payoff = self.training_step(optimizer)
            self.Memory.average_train_payoffs.append(average_payoff)

            self.Memory.single_train_durations.append(time.time() - m_th_iteration_start_time)
            self.Memory.train_durations_per_validation[-1] += time.time() - m_th_iteration_start_time

            if self.do_lr_decay:
                scheduler.step()

            # validation
            if m % self.validation_frequency == 0 and m > 0:
                val_start = time.time()

                cont_payoff, disc_payoff, stopping_times = self.validate(val_paths)
                log.info(
                    "After \t%s iterations the continuous value is\t %s and the discrete value is \t%s" % (m + len(self.Memory.average_pretrain_payoffs), round(cont_payoff, 3), round(disc_payoff, 3)))

                self.ProminentResults.process_current_iteration(self, m + len(self.Memory.average_pretrain_payoffs), cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

                self.Memory.val_continuous_value_list.append(cont_payoff)
                self.Memory.val_discrete_value_list.append(disc_payoff)

                self.Memory.val_durations.append(time.time() - val_start)

                i_value = [max(s*range(0, self.N+1)) for s in stopping_times]
                self.Memory.average_val_stopping_time.append(np.mean(i_value))

                self.Memory.train_durations_per_validation.append(0)
                self.Memory.total_net_durations_per_validation.append(0)

            # reset nets
            if m % (5*self.validation_frequency) == 0 and m > 5 and (self.algorithm == 6 or self.algorithm == 7):
                summed_stopping_times = np.sum(stopping_times, axis=0)
                while summed_stopping_times[-1] == 0:
                    summed_stopping_times = summed_stopping_times[:-1]
                indicies_to_reset = np.where(summed_stopping_times < 2)[0]  # [0] so it works...
                self.Memory.net_resets += "(" + str(m) + ") " + str(indicies_to_reset) + "\t\t"
                if len(indicies_to_reset) != 0:
                    for i in indicies_to_reset:
                        if i < len(self.u):
                            self.u[i] = self.define_net_with_path_dim_k(self.path_dim[i])
                            self.empty_pretrain_net_n(i, self.T_max/8, max(m, 100), end_condition=min(m/10, 30))

                            params = []
                            for k in range(len(self.u)):
                                params += list(self.u[k].parameters())
                            optimizer = self.return_optimizer(params)
                            if self.do_lr_decay:
                                scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
                                scheduler.verbose = False  # prints updates
                        else:
                            print("This shouldn't happen")

            m += 1

        self.Memory.train_durations_per_validation.pop()
        self.Memory.total_net_durations_per_validation.pop()

        # set final prominent result
        self.ProminentResults.set_final_net(self, m-1 + len(self.Memory.average_pretrain_payoffs), cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

        return m + len(self.Memory.average_pretrain_payoffs), self.ProminentResults, self.Memory

    def pretrain(self, duration, iterations):
        self.Memory.total_net_durations_per_validation.append(0)  # note: wird später in andere variable geschrieben

        avg_list = []
        for n in range(self.N):
            avg_list.extend(self.empty_pretrain_net_n(n, duration / self.N, iterations / self.N))

        return avg_list

    # pretrain the nth net
    def empty_pretrain_net_n(self, n, max_duration, max_iterations, end_condition=8):
        start_time = time.time()

        pretrain_batch_size = int(self.batch_size * self.training_size_during_pretrain)

        net_list = [self.u[n]]

        # version 1: no empty nets after n+1
        # version 2: empty nets -> full length

        # version 1 is much faster but gets worse values.
        if self.pretrain_with_empty_nets:
            k2 = self.Model.getN() - n - 1
            for _ in range(k2):
                net_list.append(fake_net)
        else:
            k2 = 0

        params = list(net_list[0].parameters())

        optimizer = self.return_optimizer(params)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates

        avg_list = []
        m = 0
        maximum = [0, 0]  # max, iterations since max
        while (maximum[1] < end_condition and (m < max_iterations and (time.time() - start_time) / 60 < max_duration)) or m < 30:
            iteration_start = time.time()  # I found a strange time used pattern when pretraining. Find out why.
            if self.pretrain_with_empty_nets:
                training_paths = self.Model.generate_paths(pretrain_batch_size, self.antithetic_train)
            else:
                training_paths = self.Model.generate_paths(pretrain_batch_size, self.antithetic_train, N=n+1)

            if isinstance(self.Model, W_RobbinsModel.W_RobbinsModel):
                training_paths = training_paths[:, :, -k2-2:]
            else:
                for j in range(len(training_paths)):
                    if isinstance(self.Model, RobbinsModel):
                        training_paths[j] = training_paths[j][-k2-2:]  # k2 = 0 if no empty nets
                    else:
                        training_paths[j] = training_paths[j][:, -k2-2:]
            avg = self.training_step(optimizer, training_paths, net_list=net_list)
            avg_list.append(avg)

            if avg > maximum[0]:
                maximum[0] = avg
                maximum[1] = 0
            else:
                maximum[1] += 1

            if self.do_lr_decay:
                scheduler.step()

            self.Memory.single_train_durations.append(time.time() - iteration_start)
            m += 1
        print("Pretrain stops. There were " + str(maximum[1]) + " iterations since the maximum was attained. There have been " + str(m) + " Iterations and there should be no more then " +
              str(max_iterations) + " Time spend is " + str((time.time() - start_time)/60) + " and it should be less then " + str(max_duration))

        return avg_list

    # one training step
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

        for net in self.u:
            if net != fake_net:
                net.train(mode=True)

        individual_payoffs = []

        for j in range(len(training_paths)):
            h, _ = self.generate_stopping_time_factors_and_discrete_stoppoint_from_path(training_paths[j], True, net_list=net_list)
            U[j, :] = h[:, 0]
            individual_payoffs.append(self.Model.calculate_payoffs(U[j, :], training_paths[j], self.Model.getg, self.t, device=self.device))
        average_payoff = torch.sum(torch.stack(individual_payoffs)) / len(individual_payoffs)

        loss = -average_payoff

        t = time.time()
        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)  # fürs debugging
        loss.backward()
        optimizer.step()
        self.Memory.total_net_durations_per_validation[-1] += time.time() - t

        return average_payoff.item()

    # validiere/teste auf gegebenen Pfaden
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

    # Generiert die diskrete Stoppzeit aus der stetigen
    def generate_discrete_stopping_time_from_u(self, u):
        # between 0 and N
        for n in range(self.N + 1):
            if u[n] > 0.5:
                return n
        return self.N

    # Auf einem Pfad wird die absolute Wahrscheinlichkeit zu stoppenfür die diskrete und stetige Stoppzeit berechnet.
    def generate_stopping_time_factors_and_discrete_stoppoint_from_path(self, x_input, grad, net_list=None):
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
                        x.append(torch.tensor(x_input[n][:], dtype=torch.float32, requires_grad=grad, device=self.device))
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
                local_u.append(torch.ones(1, device=self.device))
            if isinstance(local_u[n], int) or isinstance(local_u[n], float):
                U.append(local_u[n] * (torch.ones(1) - probability_sum[n]))
            else:
                U.append(local_u[n] * (torch.ones(1, device=self.device) - probability_sum[n]))

        z = torch.stack(U)
        assert torch.sum(z).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(z).item())
        return z, self.generate_discrete_stopping_time_from_u(local_u)
