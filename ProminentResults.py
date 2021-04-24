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
        if val_cont_payoff > self.cont_best_result.test_cont_value or (val_cont_payoff == self.cont_best_result.test_cont_value and val_disc_payoff > self.cont_best_result.test_disc_value):
            self.cont_best_result.update(NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result)
            self.log.info("This is a new cont best!!!!!")

        if val_disc_payoff > self.disc_best_result.test_disc_value or (val_disc_payoff == self.disc_best_result.test_disc_value and val_cont_payoff > self.disc_best_result.test_cont_value):
            self.disc_best_result.update(NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result)
            self.log.info("This is a new disc best!!!!!")

    def set_final_net(self, NN, m, val_cont_payoff, val_disc_payoff, stopping_times, total_time_used):
        self.final_result.update(NN, m, val_cont_payoff, val_disc_payoff, stopping_times, total_time_used)

    def get_m_max(self):
        return max(self.cont_best_result.m, self.disc_best_result.m)

    def get_max_time_to_best_result(self):
        return max(self.cont_best_result.time_to_this_result, self.disc_best_result.time_to_this_result)

    def final_validation(self, paths_for_final_val):
        self.cont_best_result.final_validation(paths_for_final_val, self.NN)
        self.disc_best_result.final_validation(paths_for_final_val, self.NN)
        self.final_result.final_validation(paths_for_final_val, self.NN)


class IndividualBestResult:
    def __init__(self, path):
        self.test_cont_value = -1
        self.test_disc_value = -1
        self.m = 0
        self.test_stopping_times = None
        self.val_stopping_times = None
        self.time_to_this_result = None
        self.val_cont_value = None
        self.val_disc_value = None
        self.path = path

    def update(self, NN, m, cont_payoff, disc_payoff, stopping_times, time_to_this_result):
        self.m = m
        self.test_cont_value = cont_payoff
        self.test_disc_value = disc_payoff
        self.test_stopping_times = stopping_times
        self.time_to_this_result = time_to_this_result
        dictionary = {}
        for k in range(len(NN.u)):
            dictionary["Model" + str(k)] = NN.u[k].state_dict()
        torch.save(dictionary, self.path)

    def load_state_dict_into_given_net(self, NN):
        checkpoint = torch.load(self.path)
        for k in range(len(NN.u)):
            NN.u[k].load_state_dict(checkpoint["Model" + str(k)])
            NN.u[k].eval()
        return NN

    def final_validation(self, paths, NN):
        NN = self.load_state_dict_into_given_net(NN)
        cont, disc, stop = NN.validate(paths)
        self.val_cont_value = cont
        self.val_disc_value = disc
        self.val_stopping_times = stop
