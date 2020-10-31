from copy import deepcopy
# TODO: DEEEPCOPY
from Util import *


class ProminentResults:
    def __init__(self, log, NN):
        self.cont_best_result = IndividualBestResult("cont_NN.pt")
        self.disc_best_result = IndividualBestResult("disc_NN.pt")
        self.final_result = IndividualBestResult("final_NN.pt")
        self.log = log
        self.NN = NN

    def process_current_iteration(self, NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result):
        if val_cont_payoff > self.cont_best_result.cont_value or (val_cont_payoff == self.cont_best_result.cont_value and val_disc_payoff > self.cont_best_result.disc_value):
            self.cont_best_result.update(NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result)
            self.log.info("This is a new cont best!!!!!")

        if val_disc_payoff > self.disc_best_result.disc_value or (val_disc_payoff == self.disc_best_result.disc_value and val_cont_payoff > self.disc_best_result.cont_value):
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
        self.cont_value = -1
        self.disc_value = -1
        self.m = 0
        self.stopping_times = None
        self.final_stopping_times = None
        self.time_to_this_result = None
        self.final_cont_value = None
        self.final_disc_value = None
        self.path = path

    def update(self, NN, m, val_cont_payoff, val_disc_payoff, stopping_times, time_to_this_result):
        # TODO: DEEEPCOPY something else
        self.m = m
        self.cont_value = val_cont_payoff
        self.disc_value = val_disc_payoff
        self.stopping_times = stopping_times
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
        self.final_cont_value = cont
        self.final_disc_value = disc
        self.final_stopping_times = stop

    """
    self.log = log
    self.lr = config.lr  # Lernrate
    self.lr_sheduler_breakpoints = config.lr_sheduler_breakpoints
    self.nu = config.N * (2 * config.d + 1) * (config.d + 1)
    self.N = config.N
    self.d = config.d
    self.u = []
    self.Model = Model
    self.t = config.time_partition
    self.net_net_duration = []

    self.internal_neurons = config.internal_neurons
    self.activation1 = config.activation1
    self.activation2 = config.activation2
    self.optimizer = config.optimizer

    self.validation_frequency = config.validation_frequency
    self.antithetic_variables = config.antithetic_variables

    self.out = out
    """
