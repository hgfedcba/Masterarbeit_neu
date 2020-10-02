from Util import *
from NetDefinitions import activation_functions
import torch
import torch.optim as optim

class NetConfig:
    def __init__(self, log):
        self.algorithm = 0  # 0 is source, 1 is mine

        self.internal_neurons = 50
        self.activation1 = torch.tanh
        # self.activation1 = torch.nn.functional.selu
        # self.activation1 = torch.nn.SELU()
        self.activation2 = torch.sigmoid
        self.optimizer = optim.Adam

        self.validation_frequency = 2
        self.antithetic_variables = True  # only in validation!

        self.pretrain = True
        self.pretrain_func = self.am_put_default_pretrain
        self.pretrain_iterations = 800

        self.stop_paths_in_plot = True  # TODO:use

        self.max_number_iterations = 5001
        self.max_minutes_for_iteration = 50
        self.batch_size = 32
        self.val_size = 64
        self.final_val_size = 126