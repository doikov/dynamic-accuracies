import numpy as np
import os
from experimental_tools import *
from newton_methods import cubic_newton, contracting_cubic_newton
from oracles import PDifferenceOracle
from utils import get_tolerance, get_tolerance_strategy

def run_experiment(n, line_search, max_iters):    
    print('Experiment: \t n = %d, \t line_search = %s, \t max_iters = %d.' % 
          (n, str(line_search), max_iters))

    oracle = PDifferenceOracle(3)
    x_star = np.zeros(n)
    f_star = oracle.func(x_star)
    x_0 = np.ones(n)

    H_0 = 1.0
    tolerance = get_tolerance({'criterion': 'func',
                               'f_star': f_star,
                               'tolerance': 1e-9})

    power_strategy_1 = get_tolerance_strategy({'strategy': 'power', 
                                               'c': 1.0, 
                                               'alpha': 1, 
                                               'label': r'$1/k$'})
    power_strategy_3 = get_tolerance_strategy({'strategy': 'power', 
                                               'c': 1.0, 
                                               'alpha': 3, 
                                               'label': r'$1/k^3$'})
    adaptive_strategy = get_tolerance_strategy({'strategy': 'adaptive',
                                                'c': 1.0,
                                                'alpha': 1,
                                                'label': 'adaptive'})
    
    subsolver = 'NCG'
    stopping_criterion_inner = 'grad_uniform_convex'

    histories = []
    labels = []

    _, status, history_CN_power = \
        cubic_newton(oracle, x_0, tolerance, 
                     max_iters=max_iters, 
                     H_0=H_0, 
                     line_search=line_search, 
                     inner_tolerance_strategy=power_strategy_3, 
                     subsolver=subsolver, 
                     trace=True,
                     stopping_criterion_subproblem=stopping_criterion_inner)
    histories.append(history_CN_power)
    labels.append(r'CN, $1/k^3$')

    _, status, history_CN_adaptive = \
        cubic_newton(oracle, x_0, tolerance, 
                     max_iters=max_iters, 
                     H_0=H_0, 
                     line_search=line_search, 
                     inner_tolerance_strategy=adaptive_strategy, 
                     subsolver=subsolver, 
                     trace=True,
                     stopping_criterion_subproblem=stopping_criterion_inner)
    histories.append(history_CN_adaptive)
    labels.append('CN, adaptive')

    _, status, history_CN_averaging = \
        cubic_newton(oracle, x_0, tolerance, 
                     max_iters=max_iters, 
                     H_0=H_0, 
                     line_search=line_search, 
                     inner_tolerance_strategy=power_strategy_3, 
                     subsolver=subsolver, 
                     trace=True,
                     stopping_criterion_subproblem=stopping_criterion_inner,
                     averaging=True)
    histories.append(history_CN_averaging)
    labels.append(r'Averaging, $1/k^3$')

    if not line_search:
        _, status, history_CN_contracting = \
            contracting_cubic_newton(oracle, x_0, tolerance, 
                                     max_iters=max_iters, 
                                     H_0=H_0, 
                                     prox_steps_tolerance_strategy=
                                     power_strategy_1,
                                     newton_steps_tolerance_strategy=
                                     power_strategy_1,
                                     trace=True)
        histories.append(history_CN_contracting)
        labels.append(r'Contracting')

    filename = os.getcwd() + '/plots/averaging_%d' % n
    title = r'$n = %d$' % n
    if line_search:
      filename += '_ls'
      title += ', line search'
    
    plot_func_residual(histories, None, f_star, labels, 
                       ['blue', 'red', 'tab:green', 'tab:purple'], 
                       ['-.', '-', '-', ':'], 
                       [3, 2, 5, 5], 
                       [0.6, 1, 0.8, 0.8], 
                       title, 
                       'Iterations', 
                       filename=filename+'.pdf',
                       figsize=(5.5, 5))

run_experiment(n=50, line_search=False, max_iters=400)
run_experiment(n=100, line_search=False, max_iters=400)
run_experiment(n=200, line_search=False, max_iters=400)

run_experiment(n=50, line_search=True, max_iters=400)
run_experiment(n=100, line_search=True, max_iters=400)
run_experiment(n=200, line_search=True, max_iters=400)
