import math
import time
from ModelDefinitions import add_mu_c_x, add_sigma_c_x, add_american_put, add_bermudan_max_call, binomial_trees
from ModelDefinitions import mu_dict, payoff_dict, sigma_dict

from MarkovBlackScholesModel import MarkovBlackScholesModel

from NetDefinitions import add_am_call_default_pretrain, add_am_put_default_pretrain, add_multiplicative_lr_scheduler, pretrain_functions, lr_decay_algs, optimizers, add_step_lr_scheduler
from NetDefinitions import Adam, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid, tanh, sigmoid, id

from sklearn.model_selection import ParameterGrid

from Config import Config

import numpy as np
import matplotlib.pyplot as plt

import NN
import copy

from Util import mylog
import time

from Memory import Memory as MemeClass

import Out


def main5():
    # TO DO: freeze t  !!!!!
    # TO DO: Net

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
    test_size = 256
    val_size = 2048

    """
    # using 1 as an actual benchmark
    # Make new version with actual N
    max_minutes = 20
    batch_size = 512
    test_size = 1024
    val_size = 8192
    """
    x_plot_range_for_net_plot = [10, 50]

    Model = MarkovBlackScholesModel(T, N, d, K, delta, mu, sigma, g, xi)
    # Model.set_reference_value(binomial_trees(xi, r, sigma_constant, T, 2000, K))
    Model.set_reference_value(6.245555049146182)  # N = 2000
    Model.update_parameter_string()

    test_paths_file = "../test_paths_1.npy"
    val_paths_file = "../val_paths_1.npy"
    test_paths = np.load(test_paths_file, mmap_mode="r")
    val_paths = np.load(val_paths_file, mmap_mode="r")

    # erster Test mit t=5

    class UserDefinedObjective(object):
        def calc_ders_multi(self, approx, target, weight):
            local_N = approx.__len__()
            sum = []
            U = []
            approxes[local_N-1] = 1
            for n in range(local_N):
                if n > 0:
                    sum.append(sum[n - 1] + U[n - 1])  # 0...n-1
                else:
                    sum.append(0)
                U.append(approx[n] * (1 - sum[n]))

            def f_prime(x):
                return - (K > x) * math.exp(-r*5)

            def f_prime_prime(x):
                return 0

            der1 = []
            der2 = []
            for n in range(local_N):
                der1.append(U[n]*f_prime(target[n]))
                der2.append(0)

            return der1, der2

    from catboost import Pool, CatBoostRegressor, CatBoostClassifier

    # model = CatBoostRegressor(depth=2, loss_function=UserDefinedObjective())
    model = CatBoostClassifier(iterations=10,
                               learning_rate=1,
                               depth=2,
                               loss_function='MultiClass')
    train_data = test_paths[:test_size]
    train_data = train_data[:, 0, :]
    train_data_copy = copy.deepcopy(train_data)

    train_dataset = Pool(data=train_data, label=train_data_copy)

    model.fit(train_dataset, verbose=False)

    eval_data = val_paths[:val_size]
    eval_data = eval_data[:, 0, :]

    # Get predictions
    preds = model.predict(eval_data)

    print(preds)


    """
    f = lambda x, t: max(K-x, 0) * math.exp(-r*t)
    df_by_dx = lambda x, t: - (K > x) * math.exp(-r*t)

    class UserDefinedMultiClassObjective(object):
        def calc_ders_multi(self, approxes, target, weight):
            # approxes - indexed container of floats with predictions
            #            for each dimension of single object
            # target - contains a single expected value
            # weight - contains weight of the object
            #
            # This function should return a tuple (der1, der2), where
            # - der1 is a list-like object of first derivatives of the loss function with respect
            # to the predicted value for each dimension.
            # - der2 is a matrix of second derivatives.
            pass
    """


# def multiclass():
def main():
    from catboost import Pool, CatBoostClassifier

    test_paths_file = "../test_paths_1.npy"
    val_paths_file = "../val_paths_1.npy"
    test_paths = np.load(test_paths_file, mmap_mode="r")
    val_paths = np.load(val_paths_file, mmap_mode="r")

    train_data = test_paths[:4, 0, :]
    eval_data = val_paths[:4, 0, :]
    """
    train_data = [["summer", 1924, 44],
                  ["summer", 1932, 37],
                  ["winter", 1980, 37],
                  ["summer", 2012, 204]]

    eval_data = [["winter", 1996, 197],
                 ["winter", 1968, 37],
                 ["summer", 2002, 77],
                 ["summer", 1948, 59]]

    cat_features = [0]

    train_label = ["France", "USA", "USA", "UK"]
    eval_label = ["USA", "France", "USA", "UK"]
    """

    train_label = [1, 2, 3, 4]
    eval_label = [5, 3, 2, 1]

    train_dataset = Pool(data=train_data,
                         label=train_label,
                         # cat_features=cat_features)
                         )

    eval_dataset = Pool(data=eval_data,
                        label=eval_label,
                        #cat_features=cat_features
                        )

    class UserDefinedObjective(object):
        def calc_ders_multi(self, approx, target, weight):
            approx = np.array(approx) - max(approx)
            exp_approx = np.exp(approx)
            exp_sum = exp_approx.sum()
            grad = []
            hess = []
            for j in range(len(approx)):
                der1 = -exp_approx[j] / exp_sum
                if j == target:
                    der1 += 1
                hess_row = []
                for j2 in range(len(approx)):
                    der2 = exp_approx[j] * exp_approx[j2] / (exp_sum ** 2)
                    if j2 == j:
                        der2 -= exp_approx[j] / exp_sum
                    hess_row.append(der2 * weight)

                grad.append(der1 * weight)
                hess.append(hess_row)

            return (grad, hess)

        def calc_ders_range(self, approx, target, weight):
            approx = np.array(approx) - max(approx)
            exp_approx = np.exp(approx)
            exp_sum = exp_approx.sum()
            grad = []
            hess = []
            for j in range(len(approx)):
                der1 = -exp_approx[j] / exp_sum
                if j == target:
                    der1 += 1
                hess_row = []
                for j2 in range(len(approx)):
                    der2 = exp_approx[j] * exp_approx[j2] / (exp_sum ** 2)
                    if j2 == j:
                        der2 -= exp_approx[j] / exp_sum
                    hess_row.append(der2 * weight)

                grad.append(der1 * weight)
                hess.append(hess_row)

            return (grad, hess)

    class CustomMetric:
        def is_max_optimal(self):
            return True  # greater is better

        def evaluate(self, approxes, target, weight):
            return approxes, 0

        def get_final_error(self, error, weight):
            return error

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=10,
                               learning_rate=1,
                               depth=2,
                               loss_function=UserDefinedObjective,
                               eval_metric=CustomMetric)
    # Fit model
    model.fit(train_dataset)
    """
    # Get predicted classes
    preds_class = model.predict(eval_dataset)
    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(eval_dataset)
    # Get predicted RawFormulaVal
    preds_raw = model.predict(eval_dataset,
                              prediction_type='RawFormulaVal')
    """
    assert True


def custom_metric():
    from catboost import CatBoostClassifier, Pool

    train_data = [[0, 3],
                  [4, 1],
                  [8, 1],
                  [9, 1]]

    train_labels = [0, 0, 1, 1]

    eval_data = [[2, 1],
                 [3, 1],
                 [9, 0],
                 [5, 3]]

    eval_labels = [0, 1, 1, 0]

    eval_dataset = Pool(eval_data,
                        eval_labels)

    model = CatBoostClassifier(learning_rate=0.03,
                               custom_metric=['Logloss',
                                              'AUC:hints=skip_train~false'])

    model.fit(train_data,
              train_labels,
              eval_set=eval_dataset,
              verbose=False)

    print(model.get_best_score())


def example_gpu():
    from catboost import CatBoostClassifier

    train_data = [[0, 3],
                  [4, 1],
                  [8, 1],
                  [9, 1]]
    train_labels = [0, 0, 1, 1]

    eval_data = [[2, 4],
                 [1, 4],
                 [20, 5],
                 [10, 1]]

    model = CatBoostClassifier(iterations=1000,
                               task_type="GPU",
                               devices='0:1')
    model.fit(train_data,
              train_labels,
              verbose=False)

    # Get predictions
    preds = model.predict(eval_data)

    print(preds)




