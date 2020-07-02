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
    line_search = True
    tolerance = get_tolerance({'criterion': 'func',
                               'f_star': f_star,
                               'tolerance': 1e-8})
    subsolver = 'FGM'
    stopping_criterion_subproblem = 'func'

    constant_strategies = get_constant_strategies()
    power_strategies = get_power_strategies()
    adaptive_strategy = get_tolerance_strategy({'strategy': 'adaptive',
                                                'c': 1.0,
                                                'alpha': 1,
                                                'label': 'adaptive'})
    adaptive_15_strategy = get_tolerance_strategy({'strategy': 'adaptive',
                                                'c': 1.0,
                                                'alpha': 1.5,
                                                'label': r'adaptive $1.5$'})
    adaptive_2_strategy = get_tolerance_strategy({'strategy': 'adaptive',
                                                'c': 1.0,
                                                'alpha': 2,
                                                'label': r'adaptive $2$'})

    strategies_1 = constant_strategies
    strategies_2 = power_strategies + [constant_strategies[-1]]
    strategies_3 = [adaptive_strategy, adaptive_15_strategy, 
                    adaptive_2_strategy, constant_strategies[-1]]

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
    mu_str = ('%g' % mu)[2:]
    filename = os.getcwd() + '/plots/exact_logsumexp_%d_%s' % (n, mu_str)

    labels_1 = get_labels(strategies_1)
    histories_1 = run_method(method, strategies_1, labels_1)
    plot_func_residual_iter(histories_1, 'hess_vec_calls', f_star, labels_1, 
                       ['grey', 'grey', 'grey', 'grey'], 
                       ['-', '--', '-.', ':'], 
                       [5, 4, 3, 4], 
                       [1, 1, 1, 1], 
                       r'Log-sum-exp, $\mu = %g$: constant strategies' % mu, 
                       'Hessian-vector products', 
                       filename=filename+'_const.pdf')
    labels_2 = get_labels(strategies_2)
    histories_2 = run_method(method, strategies_2, labels_2)
    plot_func_residual_iter(histories_2, 'hess_vec_calls', f_star, labels_2, 
                       ['blue', 'blue', 'blue', 'blue', 'gray'], 
                       ['-', '--', '-.', ':', ':'], 
                       [5, 4, 3, 2, 4], 
                       [0.6, 0.6, 0.6, 0.6, 0.8], 
                       r'Log-sum-exp, $\mu = %g$: dynamic strategies' % mu, 
                       'Hessian-vector products', 
                       filename=filename+'_power.pdf')
    labels_3 = get_labels(strategies_3)
    histories_3 = run_method(method, strategies_3, labels_3)
    plot_func_residual_iter(histories_3, 'hess_vec_calls', f_star, labels_3, 
                            ['red', 'tab:orange', 'tab:orange', 'gray'], 
                            ['-', '--', '-.', ':'], 
                            [2, 4, 2, 4], 
                            [1, 1, 1, 0.8], 
                            r'Log-sum-exp, $\mu = %g$: adaptive strategies' 
                                % mu, 
                            'Hessian-vector products',
                            filename=filename+'_adaptive.pdf')

run_experiment(n=100, mu=0.1, max_iters=200)
run_experiment(n=100, mu=0.05, max_iters=500)
run_experiment(n=200, mu=0.1, max_iters=200)
run_experiment(n=200, mu=0.05, max_iters=500)
