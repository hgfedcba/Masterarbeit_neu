from Util import *
from ProminentResults import ProminentResults

class NN:
    def __init__(self, Config, Model, Durations, log):
        self.Durations = Durations
        self.ProminentResults = ProminentResults(log)

    def define_nets(self):
        assert True

    def optimization(self):
        self.ProminentResults.process_current_iteration(None, 42, 1.7, 500, None, 5)
        self.ProminentResults.process_current_iteration(None, 40, 1.9, 300, None, 6)
        self.ProminentResults.set_final_result(None, 44, 1.7, 500, None, 5)
        self.Durations.train_durations.append(5)
        self.Durations.val_durations.append(17)
        self.Durations.total_net_durations.append(23)

        return self.ProminentResults, self.Durations

    def pretrain(self):
        assert True

    def train(self):
        assert True

    def validate(self):
        assert True
