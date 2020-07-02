import math
import numpy as np
import scipy

from collections import defaultdict
from datetime import datetime

from oracles import ContractingOracle, OracleCallsCounter
from utils import get_tolerance, get_tolerance_strategy, norms_init


def cubic_newton_step_fgm(matvec, grad_k, H_k, x_0, tolerance, 
                          max_iters=1000, trace=True,
                          B=None, Binv=None):
    """
    Run Fast Gradient Method with restarts to solve the cubic subproblem:
        f(x) = <g, x> + 1/2 * <Ax, x> + H/3 * ||x||^3.
    """
    if H_k < 1e-9:
        # Warning: H_k is small.
        print('W-H', flush=True, end=' ')

    l2_norm_sqr, dual_norm_sqr, to_dual, precond = norms_init(B, Binv)
    history = defaultdict(list) if trace else None
    
    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    mat_x_k = matvec(x_k)
    x_k_norm = l2_norm_sqr(x_k) ** 0.5
    g_cur = mat_x_k + grad_k
    G_cur = g_cur + H_k * x_k_norm * to_dual(x_k)
    G_0_norm_sqr = dual_norm_sqr(G_cur)

    f_cur = 0.5 * mat_x_k.dot(x_k) + grad_k.dot(x_k)
    # Outer loop objective value.
    F_cur = f_cur + H_k * x_k_norm ** 3.0 / 3.0

    x_star = np.copy(x_0)
    F_star = F_cur
    if trace:
        history['func'].append(F_cur)
        history['grad_sqr_norm'].append(G_0_norm_sqr)

    A_k = 0.0
    L_k = 1.0
    n_inner_iters = 1

    inner_line_search = True
    max_outer_iters = 20  
    n_first = 0
    total_iter = 0

    # Outer loop.
    for k in range(max_outer_iters):

        # Run FGM.
        for n in range(n_first, n_inner_iters):

            if total_iter == max_iters:
                break
            total_iter += 1
              
            # Line search for L_k.
            line_search_iter = 0
            line_search_max_iter = 20
            while(True):
                if line_search_iter == line_search_max_iter:
                    # Warning: line search iterations are exceeded.
                    print('W-L', flush=True, end=' ')
                    break

                a_k = (n + 1) / L_k
                y_k = (a_k * v_k + A_k * x_k) / (a_k + A_k)
                y_k_norm = l2_norm_sqr(y_k) ** 0.5

                mat_y_k = matvec(y_k)
                g_y_k = mat_y_k + grad_k
                f_y_k = 0.5 * mat_y_k.dot(y_k) + grad_k.dot(y_k)
                        
                # Compute proximal gradient step (prox-function is cubed norm).
                v = y_k - precond(g_y_k) / L_k
                v_norm = l2_norm_sqr(v) ** 0.5
                T = 2 * v / (1 + np.sqrt(1 + 4 * H_k / L_k * v_norm))

                T_norm = l2_norm_sqr(T) ** 0.5
                mat_T = matvec(T)
                f_T = 0.5 * mat_T.dot(T) + grad_k.dot(T)
                      
                if not inner_line_search:
                    break
                delta = T - y_k
                # Check line search condition.
                if f_T <= f_y_k + g_y_k.dot(delta) + \
                          0.5 * L_k * l2_norm_sqr(delta):
                    break
                L_k *= 2.0
                line_search_iter += 1

            if inner_line_search:
                L_k *= 0.5

            F_T = f_T + H_k * T_norm ** 3 / 3.0
            g_T = mat_T + grad_k
            G_T = g_T + H_k * T_norm * to_dual(T)
            G_T_norm_sqr = dual_norm_sqr(G_T)

            v_k = T + (A_k / a_k) * (T - x_k)
            x_k = T
            F_k = F_T
            A_k += a_k

            if trace:
                history['func'].append(F_k)
                history['grad_sqr_norm'].append(G_T_norm_sqr)

            if F_k < F_star:
                F_star = F_k
                x_star = np.copy(x_k)

                if tolerance.stopping_condition(
                        F_k, G_T_norm_sqr, 
                        g_k_sqr_bound=dual_norm_sqr(g_T),
                        T=T):
                    return x_star, F_star, "success", history

        if F_k > F_cur:
            # Continue FGM from the previous state.
            n_first = n_inner_iters
        else:
            # Start FGM from scratch.
            v_k = np.copy(x_k)
            A_k = 0.0
            F_cur = F_k
            n_first = 0

        n_inner_iters *= 2


    return x_star, F_star, "iterations_exceeded", history


def cubic_newton_step_ncg(matvec, g, H, alpha, c, x_0, tolerance, 
                          max_iters=1000, trace=True,
                          name='Dai-Yuan',
                          B=None, Binv=None):
    """
    Nonlinear Conjugate Gradients for minimizing the function:
        f(x) = <g, x> + 1/2 * <Ax, x> + H/3 * ||x||^3 + alpha/3 * ||x - c||^3
    matvec function computes Ah product.
    """
    l2_norm_sqr, dual_norm_sqr, to_dual, precond = norms_init(B, Binv)
    history = defaultdict(list) if trace else None

    n = g.shape[0]
    x_k = np.copy(x_0)
    x_k_norm = l2_norm_sqr(x_k) ** 0.5
    x_k_c_norm = l2_norm_sqr(x_k - c) ** 0.5
    A_x_k = matvec(x_k)

    f_k = g.dot(x_k) + 0.5 * A_x_k.dot(x_k) + H * x_k_norm ** 3 / 3.0 \
            + alpha * x_k_c_norm ** 3 / 3.0
    
    G_k = g + A_x_k + H * x_k_norm * to_dual(x_k) \
            + alpha * x_k_c_norm * to_dual(x_k - c)
    G_k_sqr_norm = dual_norm_sqr(G_k)
    G_0_sqr_norm = G_k_sqr_norm

    p_k = G_k

    if trace:
        history['func'].append(f_k)
        history['grad_sqr_norm'].append(G_k_sqr_norm)    
    
    for k in range(max_iters):
        
        if k % n == 0:
            # restart every n iterations.
            p_k = G_k
    
        # Exact line search, minimizing g(h) = f(x_k - h p_k).
        A_p_k = matvec(p_k)
        A_pk_pk = A_p_k.dot(p_k)
        A_pk_xk = A_p_k.dot(x_k)
        pk_pk = to_dual(p_k).dot(p_k)
        xk_c_pk = to_dual(x_k - c).dot(p_k)
        xk_pk = to_dual(x_k).dot(p_k)
        g_p_k = g.dot(p_k)
        
        if H < 1e-9 and alpha < 1e-9:
            # Quadratic function, exact minimum.
            h_k = (A_pk_xk + g_p_k) / A_pk_pk
        else:
            h = 1.0
            # 1-D Newton method.
            EPS = 1e-12
            for i in range(20):
                r = l2_norm_sqr(h * p_k - x_k) ** 0.5
                r_c = l2_norm_sqr(h * p_k - (x_k - c)) ** 0.5
                g_G =  - g_p_k - A_pk_xk \
                       + h * (A_pk_pk + H * r * pk_pk + alpha * r_c * pk_pk) \
                       - H * r * xk_pk - alpha * r_c * xk_c_pk

                if np.abs(g_G) < EPS:
                    break

                g_H = A_pk_pk + H * r * pk_pk + alpha * r_c * pk_pk \
                        + H / r * (h * pk_pk - xk_pk) ** 2 \
                        + alpha / r_c * (h * pk_pk - xk_c_pk) ** 2
                
                h = h - g_G / g_H

            h_k = h    
        
        T = x_k - h_k * p_k
        T_norm = l2_norm_sqr(T) ** 0.5
        T_c_norm = l2_norm_sqr(T - c) ** 0.5
        A_T = matvec(T)
        f_T =  g.dot(T) + 0.5 * A_T.dot(T) + H * T_norm ** 3 / 3.0 \
                + alpha * T_c_norm ** 3 / 3.0
        g_T = g + A_T
        G_T = g_T + H * T_norm * to_dual(T) \
                + alpha * T_c_norm * to_dual(T - c)
        
        if name == 'Dai-Yuan':
            beta_k = G_T.dot(G_T) / (G_T - G_k).dot(p_k)
        elif name == 'Fletcher-Rieves':
            beta_k = - G_T.dot(G_T) / G_k.dot(G_k)
        elif name == 'Polak-Ribbiere':
            beta_k = - G_T.dot(G_T - G_k) / G_k.dot(G_k)
        else:
            print("WARNING: unknown name", name)
            return x_k, f_k, "warning", history
  
        p_k = G_T - beta_k * p_k
        x_k = T
        f_k = f_T
        G_k = G_T
        G_k_sqr_norm = dual_norm_sqr(G_k)
        x_k_norm = T_norm
        x_k_c_norm = T_c_norm
        if trace:
            history['func'].append(f_k)   
            history['grad_sqr_norm'].append(G_k_sqr_norm)

        if tolerance.stopping_condition(
                f_k, G_k_sqr_norm, 
                g_k_sqr_bound=dual_norm_sqr(g_T),
                T=T):
            return x_k, f_k, "success", history

    return x_k, f_k, "iterations_exceeded", history


def cubic_newton_step(g, A, H, B=None, eps=1e-9):
    """
    Computes minimizer of the following function:
       f(x) = <g, x> + 1/2 * <Ax, x> + H/3 * ||x||^3,
    by finding the root of the equation:
       h(r) = r - ||(A + HrB)^{-1} g|| = 0.
    """
    n = g.shape[0]
    if B is None:
        B = np.eye(n)
        l2_norm_sqr = lambda x: x.dot(x)
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)

    def f(T, T_norm):
        return g.dot(T) + 0.5 * A.dot(T).dot(T) + H * T_norm ** 3 / 3.0
    
    def h(r):
        T = scipy.linalg.cho_solve(scipy.linalg.cho_factor(
                                   A + H * r * B, lower=False), -g)
        T_norm = l2_norm_sqr(T) ** 0.5
        h_r = r - T_norm
        return h_r, T_norm, T

    try:
        min_r = 0.5 * eps
        max_r = 1.0
        max_iters = 50
        # Find max_r such that h(max_r) is nonnegative.
        for i in range(max_iters):
            h_r, T_norm, T = h(max_r)
            if h_r < -eps:
                max_r *= 2
            elif -eps <= h_r <= eps:
                return T, f(T, T_norm), "success"
            else:
                break
        
        # Binary search.
        for i in range(max_iters):
            if max_r - min_r <= eps:
                return T, f(T, T_norm), "success"
            r = min_r + 0.5 * (max_r - min_r)
            h_r, T_norm, T = h(r)
            if h_r < -eps:
                min_r = r
            elif -eps <= h_r <= eps:
                return T, f(T, T_norm), "success" 
            else:
                max_r = r

    except (LinAlgError, ValueError) as e:
            return np.zeros(n), 0.0, "linalg_error"

    return np.zeros(n), 0.0, "iterations_exceeded"


def compute_cubic_step(y_k, H_k, oracle, inner_tolerance, subsolver='FGM', 
                       B=None, Binv=None):
    """
    Computes the step of Cubic Newton, using one of the subsolvers.
    """

    grad_y_k = oracle.grad(y_k)
    d_k = np.zeros_like(y_k)
    if subsolver == 'exact':
        Hess = oracle.hess(y_k)
        T_d_k, model_T, message = \
            cubic_newton_step(grad_y_k, Hess, 0.5 * H_k, B)
    else:
        hess_vec = lambda v: oracle.hess_vec(y_k, v)
        if subsolver == 'FGM':
            T_d_k, model_T, message, hist = \
                cubic_newton_step_fgm(hess_vec, grad_y_k, 0.5 * H_k, 
                                      d_k, inner_tolerance, 
                                      max_iters=5000,
                                      trace=False,
                                      B=B, Binv=Binv)
        elif subsolver == 'NCG':
            T_d_k, model_T, message, hist = \
                cubic_newton_step_ncg(hess_vec, grad_y_k, 0.5 * H_k,
                                      0.0, d_k,
                                      d_k, inner_tolerance, 
                                      max_iters=5000,
                                      trace=False,
                                      B=B, Binv=Binv)
        else:
            print("E: unknown subsolver %s." % subsolver)
            return None, None, None
    if message != "success":
        print('W: %s' % message, end=' ', flush=True)
    return y_k + T_d_k, model_T, message


def cubic_newton(oracle, x_0, tolerance, max_iters=1000, H_0=1.0, 
                 line_search=False, trace=True,
                 inner_tolerance_strategy=None,
                 subsolver='FGM',
                 B=None, Binv=None,
                 stopping_criterion_subproblem='grad_uniform_convex',
                 averaging=False):
    
    oracle = OracleCallsCounter(oracle)

    # Initialization.
    history = defaultdict(list) if trace else None
    start_timestamp = datetime.now()
    l2_norm_sqr, dual_norm_sqr, to_dual, precond = norms_init(B, Binv)

    if inner_tolerance_strategy is None:
        inner_tolerance_strategy = get_tolerance_strategy(
            {'strategy': 'constant',
             'delta': tolerance.tolerance ** 1.5})


    x_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    func_k_prev = None

    H_k = H_0

    prev_total_inner_iters = 0
    total_inner_iters = 0

    # Main loop.
    for k in range(max_iters + 1):

        if trace:
            history['func'].append(func_k)
            history['grad_sqr_norm'].append(grad_k_norm_sqr)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())
            history['H'].append(H_k)
            history['func_calls'].append(oracle.func_calls)
            history['grad_calls'].append(oracle.grad_calls)
            history['hess_calls'].append(oracle.hess_calls)
            history['hess_vec_calls'].append(oracle.hess_vec_calls)

            history['inner_iters'].append(
                total_inner_iters - prev_total_inner_iters)
            prev_total_inner_iters = total_inner_iters


        if tolerance.stopping_condition(func_k, grad_k_norm_sqr):
            message = "success"
            break

        if k == max_iters:
            message = "iterations_exceeded"
            break

        # Compute the direction.
        d_k = np.zeros_like(x_k)
        found = False
        
        inner_tolerance_value = \
            inner_tolerance_strategy.get_tolerance(k, func_k_prev, func_k)

        if averaging:
            lambda_k = (1.0 * k / (k + 1)) ** 3
            y_k = lambda_k * x_k + (1 - lambda_k) * x_0
            grad_y_k = oracle.grad(y_k)
            func_y_k = oracle.func(y_k)
        else:
            y_k = x_k
            grad_y_k = grad_k
            func_y_k = func_k

        line_search_max_iter = 30
        for i in range(line_search_max_iter + 1):
            if i == line_search_max_iter:
                message = "adaptive_iterations_exceeded"
                break

            if stopping_criterion_subproblem == 'func' or \
                    (subsolver != 'FGM' and subsolver != 'NCG'):
                Hess_y_k = oracle.hess(y_k)
                T_d_k, model_T, message = \
                    cubic_newton_step(grad_y_k, Hess_y_k, 0.5 * H_k, B)

            # Initialize the inner tolerance.
            if stopping_criterion_subproblem == 'func':
                inner_tolerance = \
                    get_tolerance({'criterion': 'func',
                                   'f_star': model_T,
                                   'tolerance': inner_tolerance_value})
            elif stopping_criterion_subproblem == 'grad_uniform_convex':
                inner_tolerance = \
                    get_tolerance({'criterion': 'grad_uniform_convex',
                                   'p': 3.0,
                                   'sigma': 0.25 * H_k,
                                   'tolerance': inner_tolerance_value})
            elif stopping_criterion_subproblem == 'grad_norm_bound':
                inner_tolerance = \
                    get_tolerance({'criterion': 'grad_norm_bound',
                                   'c': inner_tolerance_value})
            elif stopping_criterion_subproblem == 'grad_norm_by_difference' or \
                 stopping_criterion_subproblem == 'grad_norm_by_oracle_grad':

                if stopping_criterion_subproblem == 'grad_norm_by_difference':
                    lambda_bound = lambda T: l2_norm_sqr(T - y_k) ** 2
                else:
                    lambda_bound = lambda T: dual_norm_sqr(oracle.grad(T))
                inner_tolerance = \
                    get_tolerance({'criterion': 'grad_norm_lambda_bound',
                                   'lambda_bound': lambda_bound,
                                   'c': inner_tolerance_value})
            else:
                # Heuristic stopping criterion.
                inner_tolerance = \
                    get_tolerance({'criterion': 'grad',
                                   'tolerance': inner_tolerance_value})

            hess_vec = lambda v: oracle.hess_vec(y_k, v)

            if subsolver == 'FGM':
                T_d_k, model_T, message, hist = \
                    cubic_newton_step_fgm(hess_vec, grad_y_k, 0.5 * H_k, 
                                          d_k, inner_tolerance, 
                                          max_iters=5000,
                                          trace=True,
                                          B=B, Binv=Binv)
            elif subsolver == 'NCG':
                T_d_k, model_T, message, hist = \
                    cubic_newton_step_ncg(hess_vec, grad_y_k, 0.5 * H_k,
                                          0.0, np.zeros_like(grad_y_k),
                                          d_k, inner_tolerance, 
                                          max_iters=5000,
                                          trace=True,
                                          B=B, Binv=Binv)
            
            if message != "success":
                print('W: %s' % message, end=' ', flush=True)

            if subsolver == 'FGM' or subsolver == 'NCG':
                last_inner_iters = len(hist['func'])
                total_inner_iters += last_inner_iters
            

            d_k = T_d_k
            T = y_k + T_d_k
            func_T = oracle.func(T)
            grad_T = oracle.grad(T)
            grad_T_norm_sqr = dual_norm_sqr(grad_T)

            if not line_search:
                found = True
                break

            # Check condition for H_k.
            model_min = func_y_k + model_T
            if func_T <= model_min:
                found = True
                break
            H_k *= 2

        if not found:
            message = "E: step_failure : " + message
            break

        if line_search:
            H_k *= 0.5
            H_k = max(H_k, 1e-8)

        x_k = T
        grad_k = grad_T
        func_k_prev = func_k
        func_k = func_T
        grad_k_norm_sqr = grad_T_norm_sqr

    return x_k, message, history


def contracting_cubic_newton(oracle, x_0, tolerance, max_iters=1000, H_0=1.0, 
                             trace=True, prox_steps_max_iters=None,
                             prox_steps_tolerance_strategy=None,
                             newton_steps_tolerance_strategy=None,
                             B=None, Binv=None):
    """
    Accelerated Cubic Newton, using contracted proximal iterations.
    """
    oracle = OracleCallsCounter(oracle)

    # Initialization.
    history = defaultdict(list) if trace else None
    start_timestamp = datetime.now()
    l2_norm_sqr, dual_norm_sqr, to_dual, precond = norms_init(B, Binv)

    if prox_steps_tolerance_strategy is None:
        prox_steps_tolerance_strategy = get_tolerance_strategy(
            {'strategy': 'power',
             'c': 1.0,
             'alpha': 1})

    if newton_steps_tolerance_strategy is None:
        newton_steps_tolerance_strategy = get_tolerance_strategy(
            {'strategy': 'power',
             'c': 1.0,
             'alpha': 1})

    if prox_steps_max_iters is None:
        prox_steps_max_iters = 10

    x_k = np.copy(x_0)
    v_k = np.copy(x_0)
    func_k = oracle.func(x_k)
    grad_k = oracle.grad(x_k)
    grad_k_norm_sqr = dual_norm_sqr(grad_k)
    func_k_prev = None

    H_k = H_0
    A_k = 0.0

    # Main loop.
    for k in range(max_iters + 1):

        if trace:
            history['func'].append(func_k)
            history['grad_sqr_norm'].append(grad_k_norm_sqr)
            history['time'].append(
                (datetime.now() - start_timestamp).total_seconds())
            history['H'].append(H_k)
            history['func_calls'].append(oracle.func_calls)
            history['grad_calls'].append(oracle.grad_calls)
            history['hess_calls'].append(oracle.hess_calls)
            history['hess_vec_calls'].append(oracle.hess_vec_calls)

        if tolerance.stopping_condition(func_k, grad_k_norm_sqr):
            message = "success"
            break

        if k == max_iters:
            message = "iterations_exceeded"
            break

        # Choose A_k.
        A_k_new = (k + 1) ** 3.0 / H_k
        a_k_new = A_k_new - A_k

        # We minimize Contracted objective plus the Bregman divergence of d,
        # where d(x) = 1/3||x - x_0||^3.
        contracted_oracle = ContractingOracle(oracle, a_k_new, A_k, x_k)
        d = lambda x: 1.0 / 3 * l2_norm_sqr(x - x_0) ** 1.5
        d_prime = lambda x: l2_norm_sqr(x - x_0) ** 0.5 * to_dual(x - x_0)

        d_v_k = d(v_k)
        d_prime_v_k = d_prime(v_k)
        Bregman = lambda x: d(x) - d_v_k - d_prime_v_k.dot(x - v_k) 

        T = np.copy(v_k)  # Initial point.
        g_T = contracted_oracle.grad(T)
        Func_T = contracted_oracle.func(T) + Bregman(T)
        Func_T_prev = None
        
        prox_tolerance_value = \
            prox_steps_tolerance_strategy.get_tolerance(k, func_k_prev, func_k) 
        prox_steps_tolerance = \
            get_tolerance({'criterion': 'grad_uniform_convex',
                           'p': 3.0,
                           'sigma': 0.5,
                           'tolerance': prox_tolerance_value})
        
        # Iterations for computing the proximal step.
        for i in range(prox_steps_max_iters):

            hess_vec = lambda v: contracted_oracle.hess_vec(T, v)
            g = g_T - d_prime_v_k
            alpha = 1.0
            M = 1.0
            c = x_0 - T

            inner_tolerance_value = \
                newton_steps_tolerance_strategy.get_tolerance(
                    i, Func_T_prev, Func_T)
            inner_tolerance = get_tolerance(
                {'criterion': 'grad_uniform_convex',
                 'p': 3.0,
                 'sigma': 0.5 * M,
                 'tolerance': inner_tolerance_value})

            T_d_k, model_T, message, hist = \
                cubic_newton_step_ncg(hess_vec, g, M,
                                      alpha, c,
                                      np.zeros_like(x_k), 
                                      inner_tolerance, 
                                      max_iters=100,
                                      trace=True,
                                      B=B, Binv=Binv)
            if message != 'success':
                print(message, flush=True)

            T += T_d_k

            g_T = contracted_oracle.grad(T) 
            G_T = g_T + d_prime(T) - d_prime_v_k
            G_T_norm_sqr = dual_norm_sqr(G_T)
            Func_T_prev = Func_T
            Func_T = contracted_oracle.func(T) + Bregman(T)
            if prox_steps_tolerance.stopping_condition(Func_T, G_T_norm_sqr):
                break

        v_k = T
        x_k = (a_k_new * v_k + A_k * x_k) / A_k_new
        A_k = A_k_new

        func_k_prev = func_k
        func_k = oracle.func(x_k)
        grad_k = oracle.grad(x_k)
        grad_k_norm_sqr = dual_norm_sqr(grad_k)

    return x_k, message, history

