from time import time


class Durations:
    def __init__(self):
        self.start_time = time()
        self.total_net_durations = []
        self.train_durations = []
        self.val_durations = []