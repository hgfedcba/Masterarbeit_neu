from time import time


class Memory:
    def __init__(self):
        self.start_time = time()
        self.total_net_durations = []
        self.pretrain_duration = 0
        self.train_durations = []
        self.val_durations = []

        self.val_paths = []
        self.final_val_paths = []

        self.val_continuous_payoff_list = []
        self.val_discrete_payoff_list = []

        # TODO: RECALL: discrete stopping times are saved in ProminentResults