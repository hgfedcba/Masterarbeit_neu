import copy
import time

import numpy as np
import torch

import W_RobbinsModel
from Util import *
from ProminentResults import ProminentResults
import torch.nn as nn
import torch.optim as optim
import pytest
from RobbinsModel import RobbinsModel
from NN import NN
from NN import Net


def fake_net(x):
    return 0


def real_fake_net(j, N):
    def f(x):
        return x[j] > (N-j-1)/(N-j)
    return f


class Alg10_NN(NN):
    def optimization(self, val_paths, m_out):
        self.val_paths = val_paths
        self.N = self.Model.getN()
        # self.N = test_paths.shape[2]-1

        log = self.log

        if self.algorithm == 12:
            self.u = []
            for j in range(self.N):
                self.u.append(real_fake_net(j, self.N))
            m = 0
            self.Memory.total_net_durations.append(0)
            self.Memory.train_durations.append(0)
            val_start = time.time()

            cont_payoff, disc_payoff, stopping_times = self.validate(self.val_paths)
            log.info(
                "After training \t%s nets the continuous value is\t %s and the discrete value is \t%s" % (self.N - m, round(cont_payoff, 3), round(disc_payoff, 3)))

            # Prominent Result ergibt wenig sinn, da das optimale ergebnis definitiv am ende ist
            # self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

            self.Memory.val_continuous_value_list.append(cont_payoff)
            self.Memory.val_discrete_value_list.append(disc_payoff)

            self.Memory.val_durations.append(time.time() - val_start)

            i_value = [max(s * range(0, self.N + 1)) for s in stopping_times]
            self.Memory.average_val_stopping_time.append(np.mean(i_value))
        else:
            # scheduler = None
            duration = self.T_max/self.N
            iterations = self.M_max/4
            '''
            for k in range(len(self.u)):
                # TODO: change1
                self.u[k] = fake_net
                # self.u[k] = Net(self.path_dim[k], self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, self.Model.getK())
            '''
            self.u = []
            for m in range(self.N-1, -1, -1):
                self.Memory.total_net_durations.append(0)

                m_th_iteration_start_time = time.time()

                avg_list = self.training_caller(m, duration, iterations)
                self.Memory.train_durations.append(time.time() - m_th_iteration_start_time)
                self.Memory.average_train_payoffs.extend(avg_list)

                val_start = time.time()

                saved_u = self.u

                self.u = []
                for j in range(m):
                    self.u.append(fake_net)
                self.u.extend(saved_u)

                if self.sort_net_input:
                    self.sort_input_list_inplace(self.val_paths)

                cont_payoff, disc_payoff, stopping_times = self.validate(self.val_paths)
                log.info(
                    "After training \t%s nets the continuous value is\t %s and the discrete value is \t%s" % (self.N-m, round(cont_payoff, 3), round(disc_payoff, 3)))

                # Prominent Result ergibt wenig sinn, da das optimale ergebnis definitiv am ende ist
                # self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

                self.Memory.val_continuous_value_list.append(cont_payoff)
                self.Memory.val_discrete_value_list.append(disc_payoff)

                self.Memory.val_durations.append(time.time() - val_start)

                i_value = [max(s * range(0, self.N + 1)) for s in stopping_times]
                self.Memory.average_val_stopping_time.append(np.mean(i_value))

                self.u = saved_u

                # print(avg_list)
            self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

            self.ProminentResults.set_final_net(self, m - 1, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

        return m, self.ProminentResults, self.Memory

    def training_caller(self, k, duration, iterations):

        self.u.insert(0, Net(self.path_dim[k], self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, self.Model.getK(), self.device, self.dropout_rate))

        # pretrain, deprecated
        if isinstance(self.Model, RobbinsModel) and self.algorithm == 11:
            start_time = time.time()
            barrier = 0.55+(self.Model.getN()-k)/self.Model.getN()/3  # klappt nicht
            barrier = 0.65
            self.robbins_pretrain(self.u[0], k, barrier)
            self.Memory.pretrain_duration = self.Memory.pretrain_duration + time.time() - start_time

        # TODO: Stand 8.11.
        # mit leerem pretrain funktioniert es noch schlechter als ohne, also möchte ich das train net k aus alg 20 als pretrain für alg 10 verwenden um zu schauen, ob das funktioniert.
        if self.algorithm == 14:
            # avg_list = self.train_net_k(k, iterations / 2, duration / 2, fake=True)
            avg_list = self.alg20_train_net_k(k, duration / 2, iterations / 2)
            avg_list.extend(self.train_net_k(k, iterations / 2, duration / 2))
        else:
            avg_list = self.train_net_k(k, iterations, duration)

        return avg_list

    # TODO: find out why this works and the other doesn't
    def alg20_train_net_k(self, k, duration, iterations):
        start_time = time.time()
        saved_u = copy.deepcopy(self.u)
        net = self.u[0]

        saved_N = self.N
        self.N = k + 1

        self.u = []
        """
        for j in range(k + 1):
            self.u.append(fake_net)
        
        # net = Net(n + 1, self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, self.K, self.device, self.dropout_rate)
        self.u[k] = net

        params = list(self.u[k].parameters())
        """
        self.u.append(net)
        for j in range(k + 1, saved_N):
            self.u.append(fake_net)
        params = list(self.u[0].parameters())

        optimizer = self.return_optimizer(params)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates

        avg_list = []
        m = 0
        while (m < iterations and (time.time() - start_time) / 60 < duration) or m < 40:
            """
            training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train)
            for k in range(len(training_paths)):
                training_paths[k] = training_paths[k][:n + 2]
            """
            training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train)
            if not isinstance(self.Model, W_RobbinsModel.W_RobbinsModel):
                for j in range(len(training_paths)):
                    if isinstance(self.Model, RobbinsModel):
                        training_paths[j] = training_paths[j][k:]
                    else:
                        training_paths[j] = training_paths[j][:, k:]
            else:
                training_paths = training_paths[:, :, k:]
            avg = self.training_step(optimizer, training_paths)
            avg_list.append(avg)

            if self.do_lr_decay:
                scheduler.step()
            m += 1

        self.u = saved_u
        self.u[0] = net
        self.N = saved_N

        return avg_list

    def train_net_k(self, k, iterations, duration, fake=False):
        start_time = time.time()
        m = 0
        if fake:
            saved_u = copy.deepcopy(self.u)
            for j in range(len(self.u)-1):
                self.u[j+1] = fake_net
        params = list(self.u[0].parameters())
        optimizer = self.return_optimizer(params)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates
        avg_list = []
        while (m < iterations and (time.time() - start_time) / 60 < duration) or m < 20:
            training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train)
            if not isinstance(self.Model, W_RobbinsModel.W_RobbinsModel):
                for j in range(len(training_paths)):
                    if isinstance(self.Model, RobbinsModel):
                        training_paths[j] = training_paths[j][k:]
                    else:
                        training_paths[j] = training_paths[j][:, k:]
            else:
                training_paths = training_paths[:, :, k:]
            """
            for j in range(len(training_paths)):
                if isinstance(self.Model, RobbinsModel):
                    training_paths[j] = training_paths[j][k:]
                else:
                    h = training_paths[j]
                    training_paths[j] = training_paths[j][:, k:]
            """
            avg = self.training_step(optimizer, training_paths)
            avg_list.append(avg)

            if self.do_lr_decay:
                scheduler.step()
            m += 1
        if fake:
            self.u = saved_u

        return avg_list

    # Train given net to only stop when the last value is big enough
    def robbins_pretrain(self, net, k, barrier):
        params = list(net.parameters())
        optimizer = self.optimizer_number(params, lr=0.1)

        for m in range(100):
            training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train)
            # training paths (0,...,0, rand) klappen deutlich besser als vollkommen random
            training_paths = np.zeros((self.batch_size, k+1))
            training_paths[:, -1] = np.random.random(self.batch_size)
            for j in range(len(training_paths)):
                if isinstance(self.Model, RobbinsModel):
                    training_paths[j] = training_paths[j][k]
                else:
                    training_paths[j] = training_paths[j][:, k]

            targets = []
            x = []
            losses = []
            count = 0
            for j in range(len(training_paths)):
                targets.append(int(training_paths[j][-1] > barrier))
                x.append(torch.tensor(training_paths[j], dtype=torch.float32, requires_grad=True))
                losses.append(torch.abs(targets[j]-net(x[j])))
                h1 = targets[j]
                h2 = net(x[j]).item()
                h3 = losses[-1].item()
                h4 = (torch.sum(torch.stack(losses)) / len(losses)).item()
                count += int(training_paths[j][-1] > barrier)
                if m == 10:
                    assert True

            average_payoff = torch.sum(torch.stack(losses)) / len(losses)

            t = time.time()
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            average_payoff.backward()
            optimizer.step()
            print(str(m) + "\t\t" + str(average_payoff.item()))

            if average_payoff.item() < 0.05:
                break

