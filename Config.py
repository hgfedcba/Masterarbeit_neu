import math
import numpy as np
import torch
import pytest
import time
import torch.optim as optim

from Util import *

from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid, pretrain_functions, lr_decay_algs


class Config:
    # Ich gebe Standardwerte an um den Datentyp zu deklarieren. Ich möchte die Standardwerte in fast allen Fällen überschreiben.
    def __init__(self, algorithm=0, internal_neurons=50, activation1=tanh, activation2=sigmoid, optimizer=Adam, pretrain=True, pretrain_func=pretrain_functions[0], pretrain_iterations=800,
                 max_number_of_iterations=50, max_minutes_of_iterations=5, batch_size=32, initial_lr = 0.0001,do_lr_decay=False, lr_decay_alg=lr_decay_algs[0], random_seed=23343,
                 validation_frequency=2,antithetic_variables=True,val_size=64,final_val_size=128,stop_paths_in_plot=False):

        # net
        self.algorithm = algorithm  # 0 is source, 1 is mine, 2 is christensen learn f
        self.internal_neurons = internal_neurons
        self.activation1 = activation1
        self.activation2 = activation2
        self.optimizer = optimizer

        self.pretrain = pretrain
        self.pretrain_func = pretrain_func
        self.pretrain_iterations = pretrain_iterations

        self.max_number_iterations = max_number_of_iterations
        self.max_minutes_of_iteration = max_minutes_of_iterations
        self.batch_size = batch_size
        self.initial_lr = initial_lr  # lernrate
        self.do_lr_decay = do_lr_decay
        self.lr_decay_alg=lr_decay_alg

        # Meta
        self.random_seed = random_seed
        self.validation_frequency = validation_frequency
        self.antithetic_variables = antithetic_variables  # only in validation!
        self.val_size = val_size
        self.final_val_size = final_val_size

        self.stop_paths_in_plot = stop_paths_in_plot  # TODO:use

        # Das Modell kann hier eigentlich raus, das brauche ich nur in ConfigInitializer
        """
        # Model
        self.T
        self.N
        self.xi
        self.d = 1  # dimension
        self.r = 0.05  # interest rate
        self.K = 40  # strike price
        self.delta = 0  # dividend rate
        self.sigma_constant = 0.25
        self.mu_constant = self.r
        self.sigma = self.sigma_c_x
        self.mu = self.mu_c_x
        self.g = self.american_put
        """
