import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from oracles import create_log_sum_exp_zero_oracle
from utils import get_tolerance_strategy

# Uncomment to incorporate a font into the graphs:
# mpl.rcParams['pdf.fonttype'] = 42

def generate_logsumexp(n, mu):
    np.random.seed(31415)
    m = 6 * n
    A = np.random.rand(n, m) * 2 - 1
    b = np.random.rand(m) * 2 - 1
    oracle = create_log_sum_exp_zero_oracle(A.T, b, mu)
    x_star = np.zeros(n)
    f_star = oracle.func(x_star)
    B = A.dot(A.T)
    Binv = np.linalg.inv(B)
    return oracle, x_star, f_star, B, Binv

def get_constant_strategies():
    deltas = [1e-2, 1e-4, 1e-6, 1e-8]
    labels = [r'$10^{-2}$', r'$10^{-4}$', r'$10^{-6}$', r'$10^{-8}$']
    strategies = []
    for i, delta in enumerate(deltas):
        strategy = get_tolerance_strategy({'strategy': 'constant',
                                           'delta': delta,
                                           'label': labels[i]})
        strategies.append(strategy)
    return strategies

def get_beta_strategies():
    betas = [0.5, 0.1, 0.01, 0.001]
    labels = [r'$\beta = 0.5$', r'$\beta = 0.1$', 
              r'$\beta = 0.01$', r'$\beta = 0.001$']
    strategies = []
    for i, beta in enumerate(betas):
        strategy = get_tolerance_strategy({'strategy': 'constant',
                                           'delta': beta ** 2,
                                           'label': labels[i]})
        strategies.append(strategy)
    return strategies

def get_power_strategies():
    powers = [1, 2, 3, 4]
    labels = [r'$1/k$', r'$1/k^2$', r'$1/k^3$', r'$1/k^4$']
    strategies = []
    for i, power in enumerate(powers):
        strategy = get_tolerance_strategy({'strategy': 'power',
                                           'alpha': power,
                                           'c': 1.0,
                                           'label': labels[i]})
        strategies.append(strategy)
    return strategies

def get_labels(strategies):
    return [s.label for s in strategies]

def run_method(method, parameters, labels):
    print('Run:', flush=True)
    histories = []
    for i, param in enumerate(parameters):
        print(('%d\t strategy: %s\t') % (i, labels[i]), flush=True, end='')
        start_timestamp = datetime.now()
        _, status, history = method(param)
        t_secs = (datetime.now() - start_timestamp).total_seconds()
        print(('time: %.4f \t status: %s' % (t_secs, status)), flush=True)
        histories.append(history)
    print()
    return histories

def plot_func_residual(results, xparam, f_star, labels, colors, linestyles, 
                       linewidths, alphas, title, xlabel, filename=None, 
                       figsize=None):
    if figsize is None:
        figsize=(5, 5)
    plt.figure(figsize=figsize)
    for i, result in enumerate(results):
        if xparam is not None:
            plt.semilogy(result[xparam], 
                         np.array(result['func']) - f_star, 
                         label=labels[i],
                         color=colors[i],
                         linestyle=linestyles[i],
                         linewidth=linewidths[i],
                         alpha=alphas[i])
        else: 
            plt.semilogy(np.array(result['func']) - f_star, 
                         label=labels[i],
                         color=colors[i],
                         linestyle=linestyles[i],
                         linewidth=linewidths[i],
                         alpha=alphas[i])
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Functional residual', fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

def plot_func_residual_iter(results, xparam, f_star, labels, colors, linestyles, 
                            linewidths, alphas, title, xlabel, filename=None,
                            legend_outside=False):
    if not legend_outside:
        fig = plt.figure(figsize=(8, 5))
    else:
        fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for i, result in enumerate(results):
        plt.semilogy(np.array(result['func']) - f_star, 
                     label=labels[i],
                     color=colors[i],
                     linestyle=linestyles[i],
                     linewidth=linewidths[i],
                     alpha=alphas[i])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Functional residual', fontsize=18)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    for i, result in enumerate(results):
        plt.semilogy(result[xparam], 
                     np.array(result['func']) - f_star, 
                     label=labels[i],
                     color=colors[i],
                     linestyle=linestyles[i],
                     linewidth=linewidths[i],
                     alpha=alphas[i])
    plt.xlabel(xlabel, fontsize=18)
    plt.tick_params(labelsize=14)
    if not legend_outside:
        plt.legend(fontsize=14)
    else:
        plt.legend(fontsize=14, 
                   loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    st = fig.suptitle(title, fontsize=18)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    
    if filename is not None:
        plt.savefig(filename)
