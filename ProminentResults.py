from copy import deepcopy
from Util import *


class ProminentResults:
    def __init__(self, log, NN):
        self.cont_best_result = None
        self.disc_best_result = None
        self.final_result = None
        self.log = log
        self.NN = NN

        self.initialize_empty()

    def initialize_empty(self):
        self.cont_best_result = IndividualBestResult("cont_NN.pt")
        self.disc_best_result = IndividualBestResult("disc_NN.pt")
        self.final_result = IndividualBestResult("final_NN.pt")

    def process_current_iteration(self, NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result):
        if val_cont_payoff > self.cont_best_result.val_cont_value or (val_cont_payoff == self.cont_best_result.val_cont_value and val_disc_payoff > self.cont_best_result.val_disc_value):
            self.cont_best_result.update(NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result)
            self.log.info("This is a new cont best!!!!!")

        if val_disc_payoff > self.disc_best_result.val_disc_value or (val_disc_payoff == self.disc_best_result.val_disc_value and val_cont_payoff > self.disc_best_result.val_cont_value):
            self.disc_best_result.update(NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result)
            self.log.info("This is a new disc best!!!!!")

    def set_final_net(self, NN, m, val_cont_payoff, val_disc_payoff, stopping_times, total_time_used):
        self.final_result.update(NN, m, val_cont_payoff, val_disc_payoff, stopping_times, total_time_used)

    def get_m_max(self):
        return max(self.cont_best_result.m, self.disc_best_result.m)

    def get_max_time_to_best_result(self):
        return max(self.cont_best_result.time_to_this_result, self.disc_best_result.time_to_this_result)

    def test(self, paths_for_test):
        out = self.final_result.test(paths_for_test, self.NN)

        if self.final_result.m == self.cont_best_result.m:
            self.cont_best_result.test_cont_value = self.final_result.test_cont_value
            self.cont_best_result.test_disc_value = self.final_result.test_disc_value
            self.cont_best_result.test_stopping_times = self.final_result.test_stopping_times
        else:
            self.cont_best_result.test(paths_for_test, self.NN)

        if self.final_result.m == self.disc_best_result.m:
            self.disc_best_result.test_cont_value = self.final_result.test_cont_value
            self.disc_best_result.test_disc_value = self.final_result.test_disc_value
            self.disc_best_result.test_stopping_times = self.final_result.test_stopping_times
        else:
            self.disc_best_result.test(paths_for_test, self.NN)

        return out


class IndividualBestResult:
    def __init__(self, file_path):
        self.val_cont_value = -1
        self.val_disc_value = -1
        self.m = 0
        self.val_stopping_times = None
        self.test_stopping_times = None
        self.time_to_this_result = None
        self.test_cont_value = None
        self.test_disc_value = None
        self.save_file_path = file_path

    def update(self, NN, m, cont_payoff, disc_payoff, stopping_times, time_to_this_result):
        self.m = m
        self.val_cont_value = cont_payoff
        self.val_disc_value = disc_payoff
        self.val_stopping_times = stopping_times
        self.time_to_this_result = time_to_this_result
        dictionary = {}
        for k in range(len(NN.u)):
            dictionary["Model" + str(k)] = NN.u[k].state_dict()
        torch.save(dictionary, self.save_file_path)

    def load_state_dict_into_given_net(self, NN):
        checkpoint = torch.load(self.save_file_path)
        for k in range(len(NN.u)):
            NN.u[k].load_state_dict(checkpoint["Model" + str(k)])
            NN.u[k].eval()
        return NN

    def test(self, paths, NN):
        NN = self.load_state_dict_into_given_net(NN)
        cont, disc, stop = NN.validate(paths)
        self.test_cont_value = cont
        self.test_disc_value = disc
        self.test_stopping_times = stop
        return disc
