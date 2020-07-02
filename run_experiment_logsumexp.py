import numpy as np
import os
from experimental_tools import *
from newton_methods import cubic_newton
from utils import get_tolerance, get_tolerance_strategy


def run_experiment(n, mu, max_iters):    
    print('Experiment: \t n = %d, \t mu = %g, \t max_iters = %d.' % 
          (n, mu, max_iters))

    oracle, x_star, f_star, B, Binv = generate_logsumexp(n, mu)

    x_0 = np.ones(n)
    H_0 = 1.0
    line_search = False
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
                                           B=B,
                                           Binv=Binv,
                                           stopping_criterion_subproblem=
                                           stopping_criterion_subproblem)

    labels_1 = get_labels(strategies_1)
    histories_1 = run_method(method, strategies_1, labels_1)
    mu_str = ('%g' % mu)[2:]
    filename = os.getcwd() + '/plots/logsumexp_%d_%s_time' % (n, mu_str)
    plot_func_residual(histories_1, 'time', f_star, labels_1, 
                       ['grey', 'grey', 'grey', 'grey', 'red'], 
                       ['-', '--', '-.', ':', '-'], 
                       [5, 4, 3, 4, 2], 
                       [0.8, 0.8, 0.8, 0.8, 1], 
                       r'Log-sum-exp, $\mu = %g$' % mu, 
                       'Time, s', 
                       filename=filename+'_const.pdf')
    labels_2 = get_labels(strategies_2)
    histories_2 = run_method(method, strategies_2, labels_2)
    plot_func_residual(histories_2, 'time', f_star, labels_2, 
                       ['blue', 'blue', 'blue', 'blue', 'red'], 
                       ['-', '--', '-.', ':', '-'], 
                       [5, 4, 3, 2, 2], 
                       [0.6, 0.6, 0.6, 0.6, 1], 
                       r'Log-sum-exp, $\mu = %g$' % mu, 
                       'Time, s', 
                       filename=filename+'_powers.pdf')


run_experiment(n=100, mu=0.25, max_iters=2000)
run_experiment(n=100, mu=0.1, max_iters=2000)
run_experiment(n=100, mu=0.05, max_iters=10000)

