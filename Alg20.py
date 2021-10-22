import copy
from RobbinsModel import RobbinsModel
from NN import NN
from NN import Net

import W_RobbinsModel
from Util import *


def fake_net(x):
    return 0


def real_fake_net(j, N):
    def f(x):
        return x[j] > (N-j-1)/(N-j)
    return f


class Alg20_NN(NN):
    def optimization(self, val_paths, m_out):
        self.val_paths = val_paths
        self.N = self.Model.getN()

        log = self.log

        duration = self.T_max/self.N
        iterations = self.M_max/self.N
        ratio_single_to_together = 0.75

        # consists of fake nets. Fake nets are overridden gradually
        self.u = []
        for j in range(self.N):
            self.u.append(fake_net)

        for m in range(self.N):
            self.Memory.total_net_durations.append(0)

            m_th_iteration_start_time = time.time()

            avg_list = self.train_and_append_net_k(m, duration*ratio_single_to_together, iterations*ratio_single_to_together)

            # Note: last is longer
            avg_list += self.train_together(m, duration*ratio_single_to_together, iterations*ratio_single_to_together)

            self.Memory.train_durations.append(time.time() - m_th_iteration_start_time)
            self.Memory.average_train_payoffs.extend(avg_list)

            val_start = time.time()

            cont_payoff, disc_payoff, stopping_times = self.validate(self.val_paths)
            log.info(
                "After training \t%s nets the continuous value is\t %s and the discrete value is \t%s" % (m+1, round(cont_payoff, 3), round(disc_payoff, 3)))

            # Prominent Result ergibt wenig sinn, da das optimale ergebnis definitiv am ende ist
            # self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

            self.Memory.val_continuous_value_list.append(cont_payoff)
            self.Memory.val_discrete_value_list.append(disc_payoff)

            self.Memory.val_durations.append(time.time() - val_start)

            i_value = [max(s * range(0, self.N + 1)) for s in stopping_times]
            self.Memory.average_val_stopping_time.append(np.mean(i_value))

            # print(avg_list)

        self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

        self.ProminentResults.set_final_net(self, m - 1, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

        return m, self.ProminentResults, self.Memory

    # m = number of previous observations
    def train_and_append_net_k(self, n, duration, iterations):
        start_time = time.time()
        saved_u = copy.deepcopy(self.u)
        saved_N = self.N
        self.N = n+1

        self.u = []
        for j in range(n+1):
            self.u.append(fake_net)

        net = Net(n+1, self.internal_neurons, self.hidden_layer_count, self.activation_internal, self.activation_final, self.K, self.device)
        self.u[n] = net

        params = list(self.u[n].parameters())
        optimizer = self.optimizer(params, lr=self.initial_lr)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates

        avg_list = []
        m = 0
        while(m < iterations and (time.time() - start_time) / 60 < duration) or m < 20:
            training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train)
            for k in range(len(training_paths)):
                training_paths[k] = training_paths[k][:n+2]

            avg = self.training_step(optimizer, training_paths)
            avg_list.append(avg)

            if self.do_lr_decay:
                scheduler.step()
            m += 1

        self.u = saved_u
        self.u[n] = net
        self.N = saved_N

        return avg_list

    def train_together(self, n, duration, iterations):
        start_time = time.time()
        params = []
        for k in range(n+1):
            params += list(self.u[k].parameters())
        optimizer = self.optimizer(params, lr=self.initial_lr)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates
        avg_list = []
        m = 0

        while (m < iterations and (time.time() - start_time) / 60 < duration) or m < max(20, 100*(n == self.N-1)):
            avg = self.training_step(optimizer)
            avg_list.append(avg)

            if self.do_lr_decay:
                scheduler.step()
            m += 1
        return avg_list
