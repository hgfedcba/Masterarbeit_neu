from time import time


class Memory:
    def __init__(self):
        self.start_time = time()
        self.total_net_durations = []
        self.train_durations = []
        self.val_durations = []

        self.val_paths = []
        self.final_val_paths = []

        self.val_continuous_value_list = []
        self.val_discrete_value_list = []

        # TODO: RECALL: discrete stopping times are saved in ProminentResults