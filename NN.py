from Util import *
from ProminentResults import ProminentResults
import torch.nn as nn
import torch.optim as optim
import pytest

device = torch.device("cuda:0")

# TODO: IMPORTANT NOTE: I substract K from the input of the Network to ensure the learning works from the beginning on
class Net(nn.Module):
    def __init__(self, d, internal_neurons, hidden_layer_count, activation_internal, activation_final, K):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(d, internal_neurons)
        self.fcs = []
        for _ in range(hidden_layer_count):
            self.fcs.append(nn.Linear(internal_neurons, internal_neurons))
            # self.fcs.append(nn.Linear(internal_neurons, internal_neurons).cuda())
        self.fcl = nn.Linear(internal_neurons, 1)

        self.activation_internal = activation_internal
        self.activation_final = activation_final
        self.hidden_layer_count = hidden_layer_count

        self.K = K

    def forward(self, y):
        y = y-self.K
        y = self.activation_internal(self.fc1(y))
        for k in range(self.hidden_layer_count):
            y = self.activation_internal(self.fcs[k](y))
        y = self.activation_final(self.fcl(y))
        return y


class NN:
    def __init__(self, Config, Model, Memory, log, test_paths):
        self.Memory = Memory
        self.log = log

        self.test_paths = test_paths

        self.Model = Model
        self.T = Model.getT()
        self.N = Model.getN()
        self.d = Model.getd()
        self.t = Model.t
        self.K = Model.getK()

        np.random.seed(Config.random_seed)
        torch.manual_seed(Config.random_seed)

        self.T_max = Config.max_minutes_of_iteration
        self.M_max = Config.max_number_iterations

        self.initial_lr = Config.initial_lr  # Lernrate
        self.lr_decay_alg = Config.lr_decay_alg
        self.do_lr_decay = Config.do_lr_decay

        self.batch_size = Config.batch_size

        self.internal_neurons = Config.internal_neurons
        self.activation_internal = Config.activation_internal
        self.activation_final = Config.activation_final
        self.hidden_layer_count = Config.hidden_layer_count
        self.optimizer = Config.optimizer

        self.do_pretrain = Config.do_pretrain
        self.pretrain_iterations = Config.pretrain_iterations
        self.pretrain_func = Config.pretrain_func
        self.pretrain_range = Config.x_plot_range_for_net_plot

        self.validation_frequency = Config.validation_frequency
        self.antithetic_train = Config.antithetic_train

        self.algorithm = Config.algorithm
        self.u = []

        # TODO: use this
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def define_nets():
            self.u = []
            if self.algorithm == 0:
                for _ in range(self.N):
                    net = Net(self.d, self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, Model.getK())
                    # net.to(device)
                    self.u.append(net)
            elif self.algorithm == 2:
                net = Net(self.d+1, self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, Model.getK())
                self.u.append(net)

        define_nets()
        self.ProminentResults = ProminentResults(log, self)

    def optimization(self):
        log = self.log
        # scheduler = None

        params = []
        for k in range(len(self.u)):
            params += list(self.u[k].parameters())
        optimizer = self.optimizer(params, lr=self.initial_lr)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = True

        log.info("pretrain starts")
        pretrain_start = time.time()
        if self.do_pretrain:
            self.pretrain(self.pretrain_func, self.pretrain_iterations)
        self.Memory.pretrain_duration = time.time() - pretrain_start
        log.info("pretrain took \t%s seconds" % self.Memory.pretrain_duration)

        m = 0
        # continues if:
        # 1. die letzte Iteration keine validation stattgefunden hat
        # 2. die maximale zeit nicht überschritten wurde
        # 3. es keine maximale Iterationszahl gibt oder sie noch nicht erreicht wurde
        # 4. in den letzten 200 Iterationen es ein neues optimum gab
        # noch kaum iteriert wurde
        while (m % self.validation_frequency != 1 and not self.validation_frequency == 1) or \
                ((time.time() - self.Memory.start_time) / 60 < self.T_max and (self.M_max == -1 or m < self.M_max) and self.ProminentResults.get_m_max() + 200 > m)\
                or m < 10:
            self.Memory.total_net_durations.append(0)
            m_th_iteration_start_time = time.time()

            average_payoff = self.train(optimizer)
            self.Memory.train_durations.append(time.time() - m_th_iteration_start_time)

            # validation
            if m % self.validation_frequency == 0:
                val_start = time.time()

                cont_payoff, disc_payoff, stopping_times = self.validate(self.test_paths)
                log.info(
                    "After \t%s iterations the continuous value is\t %s and the discrete value is \t%s" % (m, round(cont_payoff, 3), round(disc_payoff, 3)))

                self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

                self.Memory.val_continuous_value_list.append(cont_payoff)
                self.Memory.val_discrete_value_list.append(disc_payoff)

                self.Memory.val_durations.append(time.time() - val_start)

            if self.do_lr_decay:
                scheduler.step()

            m += 1

        self.ProminentResults.set_final_net(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

        return self.ProminentResults, self.Memory

    def pretrain(self, pretrain_func, max_iterations):
        from torch.autograd import Variable

        n_sample_points = 41
        """
        x_values = np.ones((n_sample_points, self.d))
        for i in range(0, n_sample_points):
            x_values[i] = np.ones(self.d) * (self.Model.getK() + i - 16)  # True
        """
        short = self.pretrain_range
        x_values = np.reshape(np.linspace(short[0], short[1], n_sample_points), (n_sample_points, 1)) * np.ones((1, self.d))

        for m in range(len(self.u)):
            net = self.u[m]

            optimizer = optim.Adam(net.parameters(), lr=0.01)  # worked for am_put
            # optimizer = optim.Adam(net.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
            epochs = max_iterations

            def local_train():
                # net.train()  # This informs dropout and batchnorm that training is taking place. Shouldn't have any effect here.
                losses = []

                x_train = Variable(torch.from_numpy(x_values)).float()
                # x_train = x_train.to(device)
                # x_train = torch.tensor(x_values, requires_grad=True)

                y_correct = pretrain_func(x_train)

                # torch.autograd.set_detect_anomaly(True)
                for epoch in range(1, epochs):
                    loss = []
                    y_pred = []
                    if self.algorithm == 0:
                        for l in range(x_train.shape[0]):
                            y_pred.append(net(x_train[l]))
                            loss.append((y_pred[l] - y_correct[l]) ** 2)
                    elif self.algorithm == 2:
                        for n in range(self.N):
                            into = np.append(np.ones((n_sample_points, 1)) * n, x_values, 1)
                            x_train = Variable(torch.from_numpy(into)).float()
                            for l in range(x_train.shape[0]):
                                y_pred.append(net(x_train[l]))
                                loss.append((y_pred[l] - y_correct[l]) ** 2)

                    temp = sum(loss)
                    optimizer.zero_grad()
                    temp.backward()
                    optimizer.step()
                    scheduler.step()

                    losses.append(temp.item())
                    # print("epoch #", epoch)
                    # print(losses[-1])

                    if epoch == 50:
                        assert True

                    if losses[-1] < 0.1:
                        break
                    if self.algorithm == 2 and losses[-1] < 0.1*self.N:
                        break

                return losses

            # print("training start....")
            losses = local_train()

            # noinspection PyUnreachableCode
            if False:
                import matplotlib.pyplot as plt
                if m == 0:
                    draw_function(x_values, pretrain_func)
                    plt.xlabel("x")
                    plt.ylabel("target function")
                    grid(True)
                    plt.show()
                    plt.close()

                # pretrain loss
                plt.plot(range(0, losses.__len__()), losses)
                plt.xlabel("epoch")
                plt.ylabel("loss train")
                # plt.ylim([0, 100])
                grid(True)
                plt.show()
                plt.close()

                # pretrain endergebnis
                if self.algorithm == 0:
                    draw_function(x_values, self.u[m])
                elif self.algorithm == 2:
                    for k in range(self.N):
                        draw_function(x_values, self.u[0], plot_number=1+self.N+k, algorithm=2)
                plt.xlabel("x")
                plt.ylabel("u[" + str(m) + "]")
                plt.ylim([0, 1])
                grid(True)
                plt.show()
                plt.close()

    def train(self, optimizer):
        U = torch.empty(self.batch_size, self.N + 1)
        training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train)
        individual_payoffs = []

        for j in range(self.batch_size):
            h, _ = self.generate_stopping_time_factors_and_discrete_stoppoint_from_path(training_paths[j], True)
            U[j, :] = h[:, 0]
            individual_payoffs.append(self.calculate_payoffs(U[j, :], training_paths[j], self.Model.getg, self.t))
        average_payoff = torch.sum(torch.stack(individual_payoffs)) / len(individual_payoffs)

        loss = -average_payoff

        t = time.time()
        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()
        self.Memory.total_net_durations[-1] += time.time() - t

        return average_payoff.item()

    def validate(self, paths):
        L = len(paths)
        cont_individual_payoffs = []
        disc_individual_payoffs = []
        stopping_times = []

        U = torch.empty(L, self.N + 1)
        tau_list = []
        for l in range(L):
            h, tau = self.generate_stopping_time_factors_and_discrete_stoppoint_from_path(paths[l], False)
            U[l, :] = h[:, 0]
            cont_individual_payoffs.append(self.calculate_payoffs(U[l, :], paths[l], self.Model.getg, self.t))

            # for_debugging1 = paths[l]
            # for_debugging2 = h[:, 0]
            # for_debugging3 = cont_individual_payoffs[l]

            # part 2: discrete
            tau_list.append(tau)

            single_stopping_time = np.zeros(self.N + 1)
            single_stopping_time[tau_list[l]] = 1
            disc_individual_payoffs.append(self.calculate_payoffs(single_stopping_time, paths[l], self.Model.getg, self.t).item())
            # for_debugging5 = disc_individual_payoffs[-1]
            stopping_times.append(single_stopping_time)

        disc_payoff = sum(disc_individual_payoffs) / L
        temp = torch.sum(torch.stack(cont_individual_payoffs)) / L
        cont_payoff = temp.item()

        # tau list is better then stopping_times
        return cont_payoff, disc_payoff, stopping_times

    """
    def generate_discrete_stopping_time_from_U(self, U):
        # between 0 and N
        tau_set = np.zeros(self.N + 1)
        for n in range(tau_set.size):
            h1 = torch.sum(U[0:n + 1]).item()
            h2 = 1 - U[n].item()
            h3 = sum(U[0:n + 1]) >= 1 - U[n]
            tau_set[n] = torch.sum(U[0:n + 1]).item() >= 1 - U[n].item()
        tau = np.argmax(tau_set)  # argmax returns the first "True" entry
        return tau
    """

    def generate_discrete_stopping_time_from_u(self, u):
        # between 0 and N
        for n in range(self.N + 1):
            if u[n] > 0.5:
                return n
        return self.N

    # TODO: Hier geht es weiter
    def generate_stopping_time_factors_and_discrete_stoppoint_from_path(self, x_input, grad):
        local_N = x_input.shape[1]
        U = []
        sum = []
        x = []
        # x = torch.from_numpy(x_input) doesn't work for some reason

        local_u = []
        for n in range(local_N):
            if n > 0:
                sum.append(sum[n - 1] + U[n - 1])  # 0...n-1
            else:
                sum.append(0)
            if n < self.N:
                t = time.time()
                if self.algorithm == 0:
                    x.append(torch.tensor(x_input[:, n], dtype=torch.float32, requires_grad=grad))
                    # x[-1] = x[-1].to(device)
                    local_u.append(self.u[n](x[n]))
                elif self.algorithm == 2:
                    into = np.append(n+self.K, x_input[:, n])
                    x.append(torch.tensor(into, dtype=torch.float32, requires_grad=grad))
                    # x[-1] = x[-1].to(device)
                    local_u.append(self.u[0](x[n]))
                self.Memory.total_net_durations[-1] += time.time() - t
            else:
                local_u.append(torch.ones(1))
                # h[-1].to(device)
            U.append(local_u[n] * (torch.ones(1) - sum[n]))

        z = torch.stack(U)
        assert torch.sum(z).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(z).item())  # TODO: solve this better
        return z, self.generate_discrete_stopping_time_from_u(local_u)
        # return z, self.generate_discrete_stopping_time_from_U(z)

    def calculate_payoffs(self, U, x, g, t):
        assert torch.sum(torch.tensor(U)).item() == pytest.approx(1, 0.00001), "Should be 1 but is instead " + str(torch.sum(torch.tensor(U)).item())

        s = torch.zeros(1)
        for n in range(self.N + 1):
            s += U[n] * g(t[n], x[:, n])
        return s
