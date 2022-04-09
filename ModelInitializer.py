import pickle

import numpy as np

import RussianOption
from MarkovBlackScholesModel import MarkovBlackScholesModel
from ModelDefinitions import add_mu_c_x, add_sigma_c_x, add_american_put, add_russian_option, add_bermudan_max_call, binomial_trees
from NetDefinitions import add_am_call_default_pretrain, add_am_put_default_pretrain
from RobbinsModel import RobbinsModel
from Shortened_RobbinsModel import Shortened_RobbinsModel
from W_RobbinsModel import W_RobbinsModel
from Filled_RobbinsModel import Filled_RobbinsModel

# Codeausschnitte zum Speichern von Pfaden.
"""
val_paths = Model.generate_paths(1048576, True)
print("1/2")
test_paths = Model.generate_paths(1048576, True)

np.save("../val_paths_4411_5.npy", val_paths)
np.save("../test_paths_4411_5.npy", test_paths)
"""

"""
val_paths_file = "../val_paths_R60.npy"
test_paths_file = "../test_paths_R60.npy"
val_paths = Model.generate_paths(1048576, True)
print("1/2")
test_paths = Model.generate_paths(1048576, True)
with open(val_paths_file, 'wb') as f:
    pickle.dump(val_paths, f)
with open(test_paths_file, 'wb') as f:
    pickle.dump(test_paths, f)
"""


def initialize_model(option):
    angle_for_net_plot = None
    val_paths_file = None
    test_paths_file = None
    max_number = 10000
    # I deleted max_number everywhere. It was mostly 400 and was multiplied like max_time

    import pathlib
    path = pathlib.Path(__file__).parent.absolute().__str__()
    # Überprüft, an welchem Rechner ich arbeite
    if "Olus" not in path:
        main_pc = "\tZweitrechner"
    else:
        main_pc = "\tHauptrechner"

    l = False  # für einen langen test
    s = False  # für einen kurzen test
    f = False  # für den abschlusstest (nicht genutzt, sollte besonders lang sein und andere testpfade verwenden)
    vl = False  # added 11.3.2022, länger als lang
    if str(option)[-2:] == "vl":
        vl = True
        option = option[:-2]
    if str(option)[-1] == "l":
        l = True
        option = option[:-1]
    elif str(option)[-1] == "s":
        s = True
        option = option[:-1]
    elif str(option)[-1] == "f":
        f = True
        option = option[:-1]

    short = False  # verkürzte eingabedaten bei Robbins
    W = False  # gedächtnsloses Robbins Modell
    filled = False  # Für robbins mit einem netz
    if str(option)[0] == "S":
        short = True
        option = option[1:]
    elif str(option)[0] == "W":
        W = True
        option = option[1:]
    elif str(option)[0] == "0" or str(option)[0] == "F":
        filled = True
        option = option[1:]

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
        # test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 4411_2:
        # bermudan max call with 2 stocks

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
        # test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 4411_5:
        # bermudan max call with 5 stocks

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

        max_minutes = 90 * 1.5
        # batch_size = 8192
        # val_size = 8192
        # test_size = 16384

        train_size = 2048
        val_size = 4092
        test_size = 16384 * 4

        x_plot_range_for_net_plot = [60, 200]
        angle_for_net_plot = 225

        Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(36.763)
        Model.update_parameter_string(main_pc)
        val_paths_file = "../val_paths_4411_5.npy"
        test_paths_file = "../test_paths_4411_5.npy"
        val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 2:
        # Testmodell in 2 Dimensionen
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
        # test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 1:
        # Testmodell in einer Dimension
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
        train_size = 64*2
        val_size = 512*2
        test_size = 2048*2

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
        # test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == 0:
        # Testmodell, dass sehr kurz optimiert
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
        # test_paths = np.load(test_paths_file, mmap_mode="r")

    elif option == "R00":
        # Testmodell, dass Robbins sehr kurz optimiert
        N = 5
        max_minutes = 0.1
        train_size = 128
        val_size = 256
        test_size = 512
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()
        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()
        elif filled:
            Model = Filled_RobbinsModel(N)
            Model.set_reference_value()
        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

        Model.update_parameter_string(main_pc)

        val_paths = Model.generate_paths(val_size, True)
        # test_paths = Model.generate_paths(test_size, True)

    elif option == "R1" or option == "R0":
        # Längeres Robbins Testmodell
        N = 11
        train_size = 128
        val_size = 256
        test_size = 512

        if option == "R0":
            max_minutes = 0.1
        else:
            max_minutes = 10
            train_size *= 4
            val_size *= 4
            test_size *= 8
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_SR12.npy"
            test_paths_file = "../test_paths_SR12.npy"

            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_WR12.npy"
            test_paths_file = "../test_paths_WR12.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        elif filled:
            Model = Filled_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_0R12.npy"
            test_paths_file = "../test_paths_0R12.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")

        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R12.npy"
            test_paths_file = "../test_paths_R12.npy"
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        Model.update_parameter_string(main_pc)

    elif option == "R12":
        # Robbins mit 12 ZV
        N = 11
        max_minutes = 25
        train_size = 1024
        val_size = 2048
        test_size = 16384
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_SR12.npy"
            test_paths_file = "../test_paths_SR12.npy"

            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)
        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_WR12.npy"
            test_paths_file = "../test_paths_WR12.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        elif filled:
            Model = Filled_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_0R12.npy"
            test_paths_file = "../test_paths_0R12.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R12.npy"
            test_paths_file = "../test_paths_R12.npy"
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        Model.update_parameter_string(main_pc)

    elif option == "R13":
        # Robbins mit 13 ZV
        N = 12
        max_minutes = 25
        train_size = 1024
        val_size = 2048
        test_size = 16384
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_SR13.npy"
            test_paths_file = "../test_paths_SR13.npy"

            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)
        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_WR13.npy"
            test_paths_file = "../test_paths_WR13.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        elif filled:
            Model = Filled_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_0R13.npy"
            test_paths_file = "../test_paths_0R13.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")

        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R13.npy"
            test_paths_file = "../test_paths_R13.npy"
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        Model.update_parameter_string(main_pc)

    elif option == "R20":
        # Robbins mit 20 ZV
        N = 19
        max_minutes = 50  # changed 23.11. 30->50
        train_size = 1024
        val_size = 2048
        test_size = 16384
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_SR20.npy"
            test_paths_file = "../test_paths_SR20.npy"

            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_WR20.npy"
            test_paths_file = "../test_paths_WR20.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        elif filled:
            Model = Filled_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_0R20.npy"
            test_paths_file = "../test_paths_0R20.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R20.npy"
            test_paths_file = "../test_paths_R20.npy"
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        Model.update_parameter_string(main_pc)

    elif option == "R30":
        # Robbins mit 30 ZV
        N = 29
        max_minutes = 50  # changed 23.11. 30->50
        train_size = 1024
        val_size = 2048
        test_size = 16384
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_SR30.npy"
            test_paths_file = "../test_paths_SR30.npy"

            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)
        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_WR30.npy"
            test_paths_file = "../test_paths_WR30.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        elif filled:
            Model = Filled_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_0R30.npy"
            test_paths_file = "../test_paths_0R30.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R30.npy"
            test_paths_file = "../test_paths_R30.npy"
                
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        Model.update_parameter_string(main_pc)

    elif option == "R40":
        # Robbins mit 40 ZV
        N = 39
        max_minutes = 120
        train_size = 2048
        val_size = 4096
        test_size = 16384 // 2  # neccessary for secondary pc
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_SR40.npy"
            test_paths_file = "../test_paths_SR40.npy"

            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_WR40.npy"
            test_paths_file = "../test_paths_WR40.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        elif filled:
            Model = Filled_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_0R40.npy"
            test_paths_file = "../test_paths_0R40.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R40.npy"
            test_paths_file = "../test_paths_R40.npy"
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        Model.update_parameter_string(main_pc)

    elif option == "R60":
        # Robbins mit 60 ZV
        N = 59
        max_minutes = 150
        train_size = 2048
        val_size = 4096
        test_size = 8192
        x_plot_range_for_net_plot = [0, 1]
        angle_for_net_plot = 225

        if short:
            Model = Shortened_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R60.npy"
            test_paths_file = "../test_paths_R60.npy"
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            with open(test_paths_file, "rb") as fp:  # Unpickling
                test_paths = pickle.load(fp)

            val_paths = Model.convert_Robbins_paths_to_shortened_Robbins_paths(val_paths)
            test_paths = Model.convert_Robbins_paths_to_shortened_Robbins_paths(test_paths)

            with open("../val_paths_SR60.npy", 'wb') as f:
                pickle.dump(val_paths, f)
            with open("../test_paths_SR60.npy", 'wb') as f:
                pickle.dump(test_paths, f)

            val_paths_file = "../val_paths_SR60.npy"
            test_paths_file = "../test_paths_SR60.npy"

            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        elif W:
            Model = W_RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_WR60.npy"
            test_paths_file = "../test_paths_WR60.npy"

            val_paths = np.load(val_paths_file, mmap_mode="r")
            # test_paths = np.load(test_paths_file, mmap_mode="r")
        elif filled:
            Model = None
            pass
        else:
            Model = RobbinsModel(N)
            Model.set_reference_value()

            val_paths_file = "../val_paths_R60.npy"
            test_paths_file = "../test_paths_R60.npy"
            with open(val_paths_file, "rb") as fp:  # Unpickling
                val_paths = pickle.load(fp)
            # with open(test_paths_file, "rb") as fp:  # Unpickling
                # test_paths = pickle.load(fp)

        Model.update_parameter_string(main_pc)

    elif option == "Russ1":
        # Russische Option genutzt in run 42
        r = 0.05
        sigma_constant = 0.3  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 10
        N = 10  # maybe his 10=tau=T-t?
        d = 1  # dimension
        delta = 0.03  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 5*4
        train_size = 256*4
        val_size = 256*2
        test_size = 2048*8

        x_plot_range_for_net_plot = [0.25, 4]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(1.5273)
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        # test_paths = Model.generate_paths(test_size)

    elif option == "Russ11":
        # Russische Option genutzt in run 46
        r = 0.04
        sigma_constant = 0.2  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 10
        N = 10  # maybe his 10=tau=T-t?
        d = 1  # dimension
        delta = 0.04  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 5*4*8
        train_size = 256*4*4
        val_size = 256*2*2
        test_size = 2048*8*4

        x_plot_range_for_net_plot = [0.25, 4]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(-1)
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        # test_paths = Model.generate_paths(test_size)

    elif option == "Russ111":
        # Testoption
        # crashed 23.02.22
        r = 0.05
        sigma_constant = 0.3  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 10
        N = 20  # maybe his 10=tau=T-t?
        d = 1  # dimension
        delta = 0.03  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 5 * 4 * 8
        train_size = 256 * 4 * 4
        val_size = 256 * 2 * 2
        test_size = 2048 * 8 * 4

        x_plot_range_for_net_plot = [0.25, 4]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(1.5273)
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        # test_paths = Model.generate_paths(test_size)

    elif option == "Russ?":
        # Model
        r = 0.05
        sigma_constant = 0.3  # beta
        mu_constant = r
        xi = 1
        K = xi
        T = 50
        N = 20
        d = 1  # dimension
        delta = 0.03  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 0.5
        train_size = 256*2
        val_size = 256*4
        test_size = 4096*2

        x_plot_range_for_net_plot = [0.5, 3]

        Model = RussianOption.RussianOption(T, N, d, K, delta, mu, sigma, g, xi)
        Model.set_reference_value(1.4288)
        Model.update_parameter_string(main_pc)

        # val_paths_file = "../val_paths_1.npy"
        # test_paths_file = "../test_paths_1.npy"
        # val_paths = np.load(val_paths_file, mmap_mode="r")
        # test_paths = np.load(test_paths_file, mmap_mode="r")
        val_paths = Model.generate_paths(val_size)
        # test_paths = Model.generate_paths(test_size)

    elif option == "Russ0":
        # Russische testoption
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
        # test_paths = Model.generate_paths(test_size)

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
        delta = 0.03  # dividend rate
        sigma = add_sigma_c_x(sigma_constant)
        mu = add_mu_c_x(mu_constant, delta)
        g = add_russian_option(r)

        add_am_put_default_pretrain(K, 16)

        max_minutes = 30
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
        # test_paths = Model.generate_paths(test_size)
    else:
        # falsche Option eingegeben
        assert False
    if main_pc == "\tZweitrechner":
        # Der Zweitrechner hat schlechtere hardware als der hauptrechner. ich habe alle finalen tests auf dem zweitrechner durchgeführt.
        max_minutes *= 1.3
        max_number *= 1.3

    if s:
        max_minutes /= 4
        max_number /= 4

    if l:
        max_minutes *= 3
        max_number *= 3
        train_size *= 2
        val_size *= 2
        test_size *= 2

    if vl:
        max_minutes *= 6
        max_number *= 6
        train_size *= 2
        val_size *= 2
        test_size *= 2

    last_paths = False
    if f:
        max_minutes *= 4
        max_number *= 4
        train_size *= 2
        val_size *= 2
        test_size *= 8
        last_paths = True

    return val_paths, angle_for_net_plot, max_number, max_minutes, train_size, val_size, test_size, Model, x_plot_range_for_net_plot, val_paths_file, test_paths_file, last_paths


# Diese Funktion lädt die Testpfade aus der übergebenen file datei. Dafür muss sie wissen, welches Modell vorliegt und wie viele Pfaade übergeben werden sollen. last_paths ist ein boolscher Wert,
# der angibt ob die pfade von vorne oder hinten genommen werden sollen.
def load_test_paths(test_paths_file, Model, test_size, last_paths):
    if test_paths_file is None:
        test_paths = Model.generate_paths(test_size)
    else:
        if isinstance(Model, RobbinsModel):
            with open(test_paths_file, "rb") as fp:  # Unpickling
                test_paths = pickle.load(fp)
            test_size = min(test_size, len(test_paths))
        else:
            test_paths = np.load(test_paths_file, mmap_mode="r")
            test_size = min(test_size, test_paths.shape[0])
        if last_paths:
            test_paths = test_paths[-test_size:]
        else:
            test_paths = test_paths[:test_size]
    return test_paths, test_size
