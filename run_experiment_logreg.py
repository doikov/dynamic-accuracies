import numpy as np
import os
import scipy
from experimental_tools import *
from newton_methods import cubic_newton
from oracles import create_log_reg_oracle
from sklearn.datasets import load_svmlight_file
from utils import get_tolerance, get_tolerance_strategy


def run_experiment(dataset_filename, name, max_iters):    
    print('Experiment: \t %s, \t file: %s, \t max_iters = %d.' % 
          (name, dataset_filename, max_iters))

    X, y = load_svmlight_file(dataset_filename)
    oracle = create_log_reg_oracle(X, y, 1 / X.shape[0])
    x_0 = np.zeros(X.shape[1])

    print('Minimize by scipy ... ', flush=True, end='')
    f_star = \
        scipy.optimize.minimize(oracle.func, x_0, jac=oracle.grad, tol=1e-9).fun
    print('f_star = %g.' % f_star)

    H_0 = 1.0
    line_search = True
    tolerance = get_tolerance({'criterion': 'func', 
                               'f_star': f_star,
                               'tolerance': 1e-8})
    subsolver = 'FGM'
    stopping_criterion_subproblem = 'grad_uniform_convex'

    constant_strategies = get_constant_strategies()
    power_strategies = get_power_strategies()
    adaptive_strategy = get_tolerance_strategy({'strategy': 'adaptive',
                                                'c': 1.0,
                                                'alpha': 1,
                                                'label': 'adaptive'})

    strategies_1 = constant_strategies + [adaptive_strategy]
    strategies_2 = power_strategies + [adaptive_strategy]

    method = lambda strategy: cubic_newton(oracle, x_0, tolerance,
                                           max_iters=max_iters,
                                           H_0=H_0,
                                           line_search=line_search,
                                           inner_tolerance_strategy=strategy,
                                           subsolver=subsolver,
                                           trace=True,
                                           B=None,
                                           Binv=None,
                                           stopping_criterion_subproblem=
                                           stopping_criterion_subproblem)

    labels_1 = get_labels(strategies_1)
    histories_1 = run_method(method, strategies_1, labels_1)
    filename = os.getcwd() + '/plots/logreg_%s_time' % (name)
    plot_func_residual(histories_1, 'time', f_star, labels_1, 
                       ['grey', 'grey', 'grey', 'grey', 'red'], 
                       ['-', '--', '-.', ':', '-'], 
                       [5, 4, 3, 4, 2], 
                       [0.8, 0.8, 0.8, 0.8, 1], 
                       'Log-reg: %s' % name, 
                       'Time, s', 
                       filename=filename+'_const.pdf')
    labels_2 = get_labels(strategies_2)
    histories_2 = run_method(method, strategies_2, labels_2)
    plot_func_residual(histories_2, 'time', f_star, labels_2, 
                       ['blue', 'blue', 'blue', 'blue', 'red'], 
                       ['-', '--', '-.', ':', '-'], 
                       [5, 4, 3, 2, 2], 
                       [0.6, 0.6, 0.6, 0.6, 1], 
                       'Log-reg: %s' % name,
                       'Time, s', 
                       filename=filename+'_powers.pdf')



run_experiment('data/mushrooms.txt', 'mushrooms', max_iters=500)
run_experiment('data/w8a.txt', 'w8a', max_iters=200)
run_experiment('data/a8a.txt', 'a8a', max_iters=200)
