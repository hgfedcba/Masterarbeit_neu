from ModelDefinitions import add_mu_c_x, add_sigma_c_x, add_american_put, add_bermudan_max_call, binomial_trees
from ModelDefinitions import mu_dict, payoff_dict, sigma_dict
from MathematicalModel import MathematicalModel

class ConfigInitializer:
    def __init__(self, option):
        # Here i first choose the option i want to price. For every kind of option i implement a parameter grid that contains all parameters that are used for the option. I then define a model
        # class that contains all stats of the theoretical model including prices i have from other sources. i also define a new instance of the config class for every element of the parameter grid.
        # i later use this concrete config for the nets etc.

        if option == 0:
            assert True
        elif option == 4312:
            # American put in 1d
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
            mu = add_mu_c_x(mu_constant)
            g = add_american_put()

            self.Model = MathematicalModel(T, N, d, K, delta, mu, sigma, g, xi)
            self.Model.set_reference_value(binomial_trees(xi, r, sigma, T, N, K))



            # only works in 1d
            self.r = 0.06
            self.sigma_constant = 0.4  # beta
            self.mu_constant = self.r
            self.K = 40
            self.xi = 40
            self.T = 1
            self.N = 50
            self.max_number_iterations = 3000
            self.final_val_size = 4096




        #
        # TODO:different
        if self.d > 1:
            self.pretrain = False



        # TODO: better in code
        # self.preset_config_4312()
        self.preset_config_4411()