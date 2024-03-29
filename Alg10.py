import W_RobbinsModel
from Util import *
from RobbinsModel import RobbinsModel
from NN import NN


# In dieser Klasse implementiere ich die Algorithmen, die die Netze einzeln von hinten nach vorne trainieren.
class Alg10_NN(NN):
    def optimization(self, val_paths, m_out):
        self.val_paths = val_paths
        self.N = self.Model.getN()
        # self.N = test_paths.shape[2]-1

        log = self.log

        # deprectaed
        if self.algorithm == 12:
            self.u = []
            for j in range(self.N):
                self.u.append(real_fake_net(j, self.N))
            m = 0
            self.Memory.total_net_durations_per_validation.append(0)
            self.Memory.single_train_durations.append(0)
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
            if self.algorithm == 15 or self.algorithm == 16:
                duration = self.T_max/(self.N+1)
                iterations = self.M_max/(self.N+1)
            else:
                duration = self.T_max / self.N
                iterations = self.M_max / self.N

            for m in range(self.N-1, -1, -1):
                self.Memory.total_net_durations_per_validation.append(0)

                m_th_iteration_start_time = time.time()

                avg_list = self.training_caller(m, duration, iterations)
                self.Memory.train_durations_per_validation.append(time.time() - m_th_iteration_start_time)
                self.Memory.average_train_payoffs.extend(avg_list)

                val_start = time.time()

                net_list = []
                for _ in range(m):
                    net_list.append(fake_net)
                net_list.extend(self.u[m:])

                cont_payoff, disc_payoff, stopping_times = self.validate(self.val_paths, net_list=net_list)
                log.info(
                    "After training \t%s nets the continuous value is\t %s and the discrete value is \t%s" % (self.N-m, round(cont_payoff, 3), round(disc_payoff, 3)))

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

    def training_caller(self, k, duration, iterations):
        # time spendning: 1/4 pretrain, 3/4 train. Alg 15 and 16 have one extra duration-unit spend in the last run on train together
        pretrain_factor = 0.25
        if self.algorithm == 14:
            avg_list = self.train_net_k(k, iterations * pretrain_factor, duration * pretrain_factor, fake=True)
            # avg_list = self.empty_pretrain_net(self.Model.getpath_dim()[k], k, iterations / 2, duration / 2)
            avg_list.extend(self.train_net_k(k, iterations * (1-pretrain_factor), duration * (1-pretrain_factor)))

        elif self.algorithm == 15:
            avg_list = self.train_net_k(k, iterations * pretrain_factor, duration * pretrain_factor, fake=True)
            avg_list.extend(self.train_net_k(k, iterations * (1-pretrain_factor), duration * (1-pretrain_factor)))
            if k == 0:
                avg_list.extend(self.train_net_k(k, iterations, duration, train_all_nets=True))  # extra time unit

        elif self.algorithm == 16:
            avg_list = self.train_net_k(k, iterations * pretrain_factor, duration * pretrain_factor, fake=True)
            if k == 0:
                iterations *= 2.33
                duration *= 2.33
            avg_list.extend(self.train_net_k(k, iterations * (1-pretrain_factor), duration * (1-pretrain_factor)))  # extra time unit

        else:
            avg_list = self.train_net_k(k, iterations, duration)

        return avg_list

    # erklärung für den sehr merkwüurdigen value on train batch graphen: die iterationen werden sehr viel langsamer mit der zeit (höhere input dimension, mehr netzauswertungen),
    # also werden es immer weniger
    def train_net_k(self, k, iterations, duration, fake=False, train_all_nets=False):
        if fake:
            return self.empty_pretrain_net_n(k, duration, iterations)
        start_time = time.time()
        if train_all_nets:
            net_list = self.u[k:]
            params = []
            for j in range(len(net_list)):
                params += list(net_list[j].parameters())
        else:
            net_list = self.u[k:]
            params = list(net_list[0].parameters())
        optimizer = self.return_optimizer(params)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates
        avg_list = []
        m = 0
        while (m < iterations and (time.time() - start_time) / 60 < duration) or m < 20:
            iteration_start = time.time()
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
            avg = self.training_step(optimizer, training_paths, net_list=net_list)
            avg_list.append(avg)

            if self.do_lr_decay:
                scheduler.step()

            self.Memory.single_train_durations.append(time.time()-iteration_start)
            m += 1

        return avg_list
