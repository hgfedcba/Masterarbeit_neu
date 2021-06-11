from Util import *
from ProminentResults import ProminentResults
import torch.nn as nn
import torch.optim as optim
import pytest
from RobbinsModel import RobbinsModel
from NN import NN
from NN import Net


device = torch.device("cuda:0")


def fake_net(x):
    return torch.zeros(1, requires_grad=True)
    return 0


class Alg10_NN(NN):
    def optimization(self, test_paths, m_out):
        self.test_paths = test_paths
        # TODO: This doesn't work with Alg 3
        self.N = self.Model.getN()
        # self.N = test_paths.shape[2]-1

        log = self.log
        # scheduler = None
        duration = self.T_max/self.N
        iterations = self.M_max/4

        for k in range(len(self.u)):
            # TODO: change1
            self.u[k] = fake_net
            self.u[k] = Net(self.path_dim[k], self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, self.Model.getK())

        for m in range(self.N-1, -1, -1):
            self.Memory.total_net_durations.append(0)
            # TODO: 'pretrain'

            m_th_iteration_start_time = time.time()

            # TODO: change1
            # self.u[m] = Net(self.path_dim[m], self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, self.Model.getK())
            avg_list = self.train_net_k(m, duration, iterations)
            self.Memory.train_durations.append(time.time() - m_th_iteration_start_time)
            self.Memory.average_train_payoffs.extend(avg_list)

            val_start = time.time()

            cont_payoff, disc_payoff, stopping_times = self.validate(self.test_paths)
            log.info(
                "After \t%s iterations the continuous value is\t %s and the discrete value is \t%s" % (self.N-m, round(cont_payoff, 3), round(disc_payoff, 3)))

            # Prominent Result ergibt wenig sinn, da das optimale ergebnis definitiv am ende ist
            # self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

            self.Memory.val_continuous_value_list.append(cont_payoff)
            self.Memory.val_discrete_value_list.append(disc_payoff)

            self.Memory.val_durations.append(time.time() - val_start)

            i_value = [max(s * range(0, self.N + 1)) for s in stopping_times]
            self.Memory.average_test_stopping_time.append(np.mean(i_value))

        # TODO: diese Zeile ist recht sinnlos
        self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

        self.ProminentResults.set_final_net(self, m - 1, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

        return m, self.ProminentResults, self.Memory

    def train_net_k(self, k, duration, iterations):
        # TODO: get this to work
        start_time = time.time()
        m = 0
        params = list(self.u[k].parameters())

        # TODO: change 1
        params = []
        for k in range(len(self.u)):
            params += list(self.u[k].parameters())

        optimizer = self.optimizer(params, lr=self.initial_lr)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates
        avg_list = []
        while m < iterations and (time.time() - start_time) / 60 < duration:
            avg = self.train(optimizer)
            avg_list.append(avg)

            if self.do_lr_decay:
                scheduler.step()
            m += 1

        return avg_list
