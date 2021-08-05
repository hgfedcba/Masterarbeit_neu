import math
import pickle

import numpy as np

import RussianOption
from MarkovBlackScholesModel import MarkovBlackScholesModel
from ModelDefinitions import add_mu_c_x, add_sigma_c_x, add_american_put, add_russian_option, add_bermudan_max_call, binomial_trees
from NetDefinitions import add_am_call_default_pretrain, add_am_put_default_pretrain
from RobbinsModel import RobbinsModel
from W_RobbinsModel import W_RobbinsModel

"""
val_paths = Model.generate_paths(1048576, True)
test_paths = Model.generate_paths(1048576, True)

np.save("../val_paths_4411_5.npy", val_paths)
np.save("../test_paths_4411_5.npy", test_paths)
"""

def initialize_model(option):
    val_paths = None
    test_paths = None
    angle_for_net_plot = None
    max_number = 10000

    import pathlib
    path = pathlib.Path(__file__).parent.absolute().__str__()
    if not "Olus" in path:
        main_pc = "\tZweitrechner"
    else:
        main_pc = "\tHauptrechner"

    if option == 4312:
        # American put in 1d
        # This Model is stupid since it results in no sells whatsoever

        # Model
        r = 0.06
        sigma_constant = 0.4  # beta
        mu_constant = r
        K = 40
        xi = 40
        T = 1
        N = 50
        d = 1  # dimension
        delta = 0  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_american_put(d, K, r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 60

        # no bigger sizes since N=50!
        train_size = 256
        val_size = 512
        test_size = 4048
        """
        # works but takes too long
        batch_size = 1024
        val_size = 2048
        test_size = 16384
        """
        x_plot_range_for_net_plot = [10, 50]

        Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(5.318)  # verified with my binomial trees
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_4312.npy"
        test_paths_file = "../test_paths_4312.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 4411_2:
        # bermudan max call

        r = 0.05
        sigma_constant = 0.2  # beta
        mu_constant = r
        K = 100
        xi = 110
        T = 3
        N = 9
        d = 2  # dimension
        delta = 0.1  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_bermudan_max_call(K, r)

        add_am_call_default_pretrain(K + 10, 60)

        max_minutes = 60
        # batch_size = 8192
        # val_size = 8192
        # test_size = 16384

        train_size = 2048
        val_size = 4092
        test_size = 16384

        x_plot_range_for_net_plot = [30, 190]
        angle_for_net_plot = 225

        Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(21.344)
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_4411_2.npy"
        test_paths_file = "../test_paths_4411_2.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 4411_5:
        # bermudan max call

        r = 0.05
        sigma_constant = 0.2  # beta
        mu_constant = r
        K = 100
        xi = 110
        T = 3
        N = 9
        d = 5  # dimension
        delta = 0.1  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_bermudan_max_call(K, r)

        add_am_call_default_pretrain(K + 10, 60)

        max_minutes = 90
        # batch_size = 8192
        # val_size = 8192
        # test_size = 16384

        train_size = 2048
        val_size = 4092
        test_size = 16384

        x_plot_range_for_net_plot = [60, 200]
        angle_for_net_plot = 225

        Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(36.763)
        Model.update_parameter_string(main_pc)
        val_paths_file = "../val_paths_4411_5.npy"
        test_paths_file = "../test_paths_4411_5.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 2:
        # Model
        r = 0.05
        sigma_constant = 0.25  # beta
        mu_constant = r
        K = 40
        xi = 40
        T = 10
        N = 10
        d = 2  # dimension
        delta = 0  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_american_put(d, K, r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 5 / 10
        train_size = 64
        val_size = 256
        test_size = 2048

        x_plot_range_for_net_plot = [10, 50]

        Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, 200, K))
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_2.npy"
        test_paths_file = "../test_paths_2.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 1:
        # Model
        r = 0.05
        sigma_constant = 0.25  # beta
        mu_constant = r
        K = 40
        xi = 40
        T = 10
        N = 10
        d = 1  # dimension
        delta = 0  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_american_put(d, K, r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 3
        train_size = 64
        val_size = 256
        test_size = 2048

        """
        # using 1 as an actual benchmark
        # Make new version with actual N
        max_minutes = 20
        batch_size = 512
        val_size = 1024
        test_size = 8192
        """
        x_plot_range_for_net_plot = [10, 50]

        Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
        # Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, 2000, K))
        Model.set_reference_value(6.245555049146182)  # N = 2000
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_1.npy"
        test_paths_file = "../test_paths_1.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == "R1" or option == "R0":
        N = 19
        if option == "R0":
            max_minutes = 0.1
        else:
            max_minutes = 3
        train_size = 128
        val_size = 256
        test_size = 512
        x_plot_range_for_net_plot = [0, 1]

        Model = RobbinsModel(N)
        Model.set_reference_value_lower(N + 2 - 2.329)
        Model.set_reference_value_upper(N + 2 - 1.908)
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_R20.npy"
        test_paths_file = "../test_paths_R20.npy"
        with open(val_paths_file, "rb") as fp:  # Unpickling
            val_paths = pickle.load(fp)
        with open(test_paths_file, "rb") as fp:  # Unpickling
            test_paths = pickle.load(fp)

    elif option == "R2":
        N = 19
        max_minutes = 40
        max_number = 200
        train_size = 512
        val_size = 1024
        test_size = 8192
        x_plot_range_for_net_plot = [0, 1]

        Model = RobbinsModel(N)
        Model.set_reference_value_lower(N + 2 - 2.329)
        Model.set_reference_value_upper(N + 2 - 1.908)
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_R20.npy"
        test_paths_file = "../test_paths_R20.npy"
        with open(val_paths_file, "rb") as fp:  # Unpickling
            val_paths = pickle.load(fp)
        with open(test_paths_file, "rb") as fp:  # Unpickling
            test_paths = pickle.load(fp)

    elif option == "R3":
        N = 39
        max_minutes = 40
        max_number = 300
        train_size = 1024
        val_size = 2028
        test_size = 16384
        x_plot_range_for_net_plot = [0, 1]

        Model = RobbinsModel(N)
        Model.set_reference_value_lower(N + 2 - 2.329)
        Model.set_reference_value_upper(N + 2 - 1.908)
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_R40.npy"
        test_paths_file = "../test_paths_R40.npy"
        with open(val_paths_file, "rb") as fp:  # Unpickling
            val_paths = pickle.load(fp)
        with open(test_paths_file, "rb") as fp:  # Unpickling
            test_paths = pickle.load(fp)

    elif option == "RW1" or option == "RW0":
        N = 19
        if option == "RW0":
            max_minutes = 0.1
        else:
            max_minutes = 3
        train_size = 128
        val_size = 256
        test_size = 512
        x_plot_range_for_net_plot = [0, 1]

        Model = W_RobbinsModel(N)
        Model.set_reference_value(N + 2 - 2.3)
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_RW20.npy"
        test_paths_file = "../val_paths_RW20.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == "RW2":
        N = 19
        max_minutes = 20
        max_number = 200
        train_size = 512
        val_size = 1024
        test_size = 8192
        x_plot_range_for_net_plot = [0, 1]

        Model = W_RobbinsModel(N)
        Model.set_reference_value(N + 2 - 2.3)
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_RW20.npy"
        test_paths_file = "../val_paths_RW20.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == "RW3":
        N = 39
        max_minutes = 120
        max_number = 300
        train_size = 1024
        val_size = 2028
        test_size = 16384
        x_plot_range_for_net_plot = [0, 1]

        Model = W_RobbinsModel(N)
        Model.set_reference_value(N + 2 - 2.3)
        Model.update_parameter_string(main_pc)
        """
        val_paths = Model.generate_paths(1048576, True)
        test_paths = Model.generate_paths(1048576, True)

        np.save("../val_paths_RW40.npy", val_paths)
        np.save("../val_paths_RW40.npy", test_paths)
        """
        val_paths_file = "../val_paths_RW40.npy"
        test_paths_file = "../val_paths_RW40.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == "Russ1":
        # Model
        r = 0.05
        sigma_constant = 0.3  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 1
        N = 10
        d = 1  # dimension
        delta = 0  # dividend rate  # TODO: check this works
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 3
        train_size = 128
        val_size = 256
        test_size = 512

        x_plot_range_for_net_plot = [0.5, 3]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(1.2372)
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        test_paths = Model.generate_paths(test_size)

    elif option == "Russ11":
        # Model
        r = 0.05
        sigma_constant = 0.3  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 1
        N = 10
        d = 1  # dimension
        delta = 0.03  # dividend rate  # TODO: check this works
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 3
        train_size = 128
        val_size = 256
        test_size = 512

        x_plot_range_for_net_plot = [0.5, 3]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(1.2188)
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        test_paths = Model.generate_paths(test_size)

    elif option == "Russ111":
        # TODO: This is pretty stupid as it always stops at the last timestep, but it should be nice to see why my reference value is faulty
        # TODO: WORKS
        # Model
        r = 0
        sigma_constant = 0.3  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 1
        N = 10
        d = 1  # dimension
        delta = 0.05  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 0.5
        train_size = 128
        val_size = 256
        test_size = 4096

        x_plot_range_for_net_plot = [0.5, 3]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(1.2372 * math.exp(-delta * T))
        # TODO: Der japanische dude sagt das die exercise boundary der beiden optionen identisch sind und ich hoffe das bedeutet das sich beim vertauschen von delta und r sie sich nur um den diskontierungsfaktor unterscheiden
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        test_paths = Model.generate_paths(test_size)

    elif option == "Russ0":
        # Model
        r = 0.05
        sigma_constant = 0.25  # beta
        mu_constant = r
        K = 40
        xi = 40
        T = 1
        N = 10
        d = 1  # dimension
        delta = 0  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 0.5 * 0.5
        train_size = 64
        val_size = 64
        test_size = 256

        x_plot_range_for_net_plot = [20, 100]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        # Model.set_reference_value()
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        test_paths = Model.generate_paths(test_size)

    elif option == "Russ2":
        # Model
        r = 0.05
        sigma_constant = 0.3  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 1
        N = 20
        d = 1  # dimension
        delta = 0.03  # dividend rate  # TODO: check this works
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 30
        max_number = 200
        train_size = 512
        val_size = 1024
        test_size = 8192

        x_plot_range_for_net_plot = [0.5, 3]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(1.2188)
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        test_paths = Model.generate_paths(test_size)

    elif option == 0:
        # Model
        r = 0.05
        sigma_constant = 0.25  # beta
        mu_constant = r
        K = 40
        xi = 40
        T = 10
        N = 10
        d = 1  # dimension
        delta = 0  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_american_put(d, K, r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 0.5 * 0.5
        train_size = 64
        val_size = 64
        test_size = 256

        x_plot_range_for_net_plot = [10, 50]

        Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, N * 10, K))
        Model.update_parameter_string(main_pc)

        val_paths_file = "../val_paths_1.npy"
        test_paths_file = "../test_paths_1.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        test_paths = np.load(test_paths_file, mmap_mode="r")
    else:
        return 0
    if main_pc == "Zweitrechner":
        max_minutes *= 1.3
        max_number *= 1.3
    return val_paths, test_paths, angle_for_net_plot, max_number, max_minutes, train_size, val_size, test_size, Model, x_plot_range_for_net_plot