from RobbinsModel import RobbinsModel
from NN import NN
from Util import *


class Alg20_NN(NN):
    def optimization(self, val_paths, m_out):
        self.N = self.Model.getN()

        log = self.log

        div = self.N*(self.N+1)/2

        duration = self.T_max/div  # duration is proportional to number of random variables
        iterations = self.M_max/div
        ratio_single_to_together = 0.5

        # consists of fake nets. Fake nets are overridden gradually
        end = self.N

        if self.algorithm == 21:
            # I don't use the external val_paths, only their size as they are modified here
            import pickle
            if self.val_paths_file is None:
                print("generating new val paths")
                self.val_paths = self.Model.generate_paths(len(val_paths))
            elif isinstance(self.Model, RobbinsModel):
                with open(self.val_paths_file, "rb") as fp:  # Unpickling
                    self.val_paths = pickle.load(fp)
            else:
                self.val_paths = np.load(self.val_paths_file, mmap_mode="r")
            self.val_paths = self.val_paths[:len(val_paths)]

            for m in range(end):
                self.Memory.total_net_durations_per_validation.append(0)

                m_th_iteration_start_time = time.time()

                avg_list = self.empty_pretrain_net_n(m, duration * ratio_single_to_together * (m+1), iterations * ratio_single_to_together * (m+1))

                avg_list += self.train_together(m, duration * (1 - ratio_single_to_together) * (m+1), iterations * (1 - ratio_single_to_together) * (m+1), alg20=False)

                self.Memory.train_durations_per_validation.append(time.time() - m_th_iteration_start_time)
                self.Memory.average_train_payoffs.extend(avg_list)

                val_start = time.time()

                net_list = self.u[:m + 1]
                cont_payoff, disc_payoff, stopping_times = self.validate([path[:m+2] for path in self.val_paths], net_list=net_list)
                """
                log.info(
                    "After training \t%s nets the continuous value is\t %s and the discrete value is \t%s" % (m+1, round(cont_payoff, 3), round(disc_payoff, 3)))
                """

                # recall m+2 = self.N+1 = N
                l = m + 3 - robbins_problem_lower_boundary_of_W(m + 1)  # explicit threshhold function

                log.info("For N = " + str(m+2) + " the val-value is " + str(disc_payoff) + " and the reference value for W_" + str(m+2) + " is " + str(l))

                self.Memory.val_continuous_value_list.append(cont_payoff)
                self.Memory.val_discrete_value_list.append(disc_payoff)

                self.Memory.val_durations.append(time.time() - val_start)

                i_value = [max(s * range(0, m + 2)) for s in stopping_times]
                self.Memory.average_val_stopping_time.append(np.mean(i_value))

                # print(avg_list)

            self.ProminentResults.process_current_iteration(self, len(self.Memory.average_train_payoffs), cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

            self.ProminentResults.set_final_net(self, len(self.Memory.average_train_payoffs), cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

            return len(self.Memory.average_train_payoffs), self.ProminentResults, self.Memory

        elif self.algorithm == 20:
            self.val_paths = val_paths
            # recall m+2 = self.N+1 = N
            l = self.N + 2 - robbins_problem_lower_boundary_of_W(self.N)  # explicit threshhold function

            for m in range(end):
                self.Memory.total_net_durations_per_validation.append(0)

                m_th_iteration_start_time = time.time()

                avg_list = self.empty_pretrain_net_n(m, duration * ratio_single_to_together * (m+1), iterations * ratio_single_to_together * (m+1))

                avg_list += self.train_together(m, duration * (1 - ratio_single_to_together) * (m+1), iterations * (1 - ratio_single_to_together) * (m+1), alg20=True)

                self.Memory.train_durations_per_validation.append(time.time() - m_th_iteration_start_time)
                self.Memory.average_train_payoffs.extend(avg_list)

                val_start = time.time()

                net_list = self.u[:m + 1]
                for _ in range(self.Model.getN()-m-1):  # I want to understand why below it works without the -1...
                    net_list.append(fake_net)

                cont_payoff, disc_payoff, stopping_times = self.validate(self.val_paths, net_list=net_list)
                """
                log.info(
                    "After training \t%s nets the continuous value is\t %s and the discrete value is \t%s" % (m+1, round(cont_payoff, 3), round(disc_payoff, 3)))
                """
                # Prominent Result ergibt wenig sinn, da das optimale ergebnis definitiv am ende ist
                # self.ProminentResults.process_current_iteration(self, m, cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

                log.info("For N = " + str(m+2) + " the val-value is " + str(disc_payoff) + " and the reference value for W_N is " + str(l))

                self.Memory.val_continuous_value_list.append(cont_payoff)
                self.Memory.val_discrete_value_list.append(disc_payoff)

                self.Memory.val_durations.append(time.time() - val_start)

                i_value = [max(s * range(0, self.N + 1)) for s in stopping_times]
                self.Memory.average_val_stopping_time.append(np.mean(i_value))

                # print(avg_list)

            self.ProminentResults.process_current_iteration(self, len(self.Memory.average_train_payoffs), cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))

            self.ProminentResults.set_final_net(self, len(self.Memory.average_train_payoffs), cont_payoff, disc_payoff, stopping_times, (time.time() - self.Memory.start_time))
            return len(self.Memory.average_train_payoffs), self.ProminentResults, self.Memory

    def train_together(self, n, duration, iterations, alg20=True):
        start_time = time.time()
        net_list = self.u[:n+1]
        params = []
        for k in range(n+1):
            params += list(net_list[k].parameters())
        if alg20:
            for _ in range(self.Model.getN()-n):
                net_list.append(fake_net)
        optimizer = self.return_optimizer(params)
        if self.do_lr_decay:
            scheduler = self.lr_decay_alg[0](optimizer, self.lr_decay_alg[1])
            scheduler.verbose = False  # prints updates
        avg_list = []
        m = 0

        while (m < iterations and (time.time() - start_time) / 60 < duration) or m < 20:
            iteration_start = time.time()
            if alg20:
                avg = self.training_step(optimizer)
            else:
                training_paths = self.Model.generate_paths(self.batch_size, self.antithetic_train, N=n+1)

                avg = self.training_step(optimizer, training_paths, net_list=net_list)
            avg_list.append(avg)

            if self.do_lr_decay:
                scheduler.step()

            self.Memory.single_train_durations.append(time.time() - iteration_start)
            m += 1
        print("Joined training stops. There have been " + str(m) + " Iterations and there should be no more then " +
              str(iterations) + " Time spend is " + str((time.time() - start_time)/60) + " and it should be less then " + str(duration))
        return avg_list
