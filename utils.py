import numpy as np
import scipy
import warnings

def norms_init(B=None, Binv=None):
    """
    Initialize norms to use. 
    
    The primal space is the space of variables, the dual space is 
    the space of the gradients.
    Operator B transforms the primal space to the dual.
        
    l2_norm_sqr(x) := ||x||^2 = <Bx, x>, norm for the primal space;
    dual_norm_sqr(s) := ||s||^2_* = <s, B^{-1}s>, norm for the dual space;
    to_dual(x) := Bx, transforms the primal space to the dual;
    precond(s) := B^{-1}s, transforms the dual space to the primal.
    """
    if B is None:
        l2_norm_sqr = lambda x: x.dot(x)
        dual_norm_sqr = lambda x: x.dot(x)
        to_dual = lambda x: x
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)
        to_dual = lambda x: B.dot(x)
        if Binv is None:
            Binv = np.linalg.inv(B)
        dual_norm_sqr = lambda x: Binv.dot(x).dot(x)
    if Binv is None:
        precond = lambda g: g
    else:
        precond = lambda g: Binv.dot(g)

    return l2_norm_sqr, dual_norm_sqr, to_dual, precond

class Tolerance(object):
    """
    Stopping condition for optimization methods.
    """
    def __init__(self, tolerance=1e-8, criterion='grad', **kwargs):
        self.tolerance = tolerance
        self.criterion = criterion
        if self.criterion == 'func':
            self.f_star = kwargs.get('f_star', 0.0)  
        elif self.criterion == 'grad_uniform_convex':
            """
            Using the bound:
            F(x) - F^{*} <= (p - 1) / p * (1 / sigma)^{1 / (p - 1)} *
                             ||F'(x)||^{p / (p - 1)} <= tolerance.
            """
            # Degree of uniform convexity.
            self.p = kwargs.get('p', 2.0)
            # Constant of uniform convexity.
            self.sigma = kwargs.get('sigma', 1.0)
            # The constant before the gradient.
            self.alpha = (self.p - 1) / self.p * \
                         (1 / self.sigma) ** (1 / (self.p - 1))
        elif self.criterion == 'grad_norm_bound':
            """
            Checking the condition:
            ||F'(x)||^2 <= c * g_k_sqr_bound.
            """
            self.c = kwargs.get('c', 0.12 ** 2)
        elif self.criterion == 'grad_norm_lambda_bound':
            """
            Checking the condition:
            ||F'(x)||^2 <= c * lambda_bound(T)
            """
            self.c = kwargs.get('c', 0.25)
            self.lambda_bound = kwargs.get('lambda_bound', lambda x: x.dot(x))

        self.relative = kwargs.get('relative', False)
        if self.relative:
            if self.criterion == 'func':
                self.f_0 = kwargs.get('f_0', 0.0)
            if self.criterion == 'grad':
                self.g_0_sqr_norm = kwargs.get('g_0_sqr_norm', 1.0)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__
        
    def stopping_condition(self, f_k, g_k_sqr_norm, g_k_sqr_bound=None, T=None):
        if self.criterion == 'func':
            f_res = f_k - self.f_star
            if self.relative:
                return f_res <= self.tolerance * (self.f_0 - self.f_star)
            else:
                return f_res <= self.tolerance
        elif self.criterion == 'grad':
            if self.relative:
                return g_k_sqr_norm <= self.tolerance * self.g_0_sqr_norm
            else:
                return g_k_sqr_norm <= self.tolerance
        elif self.criterion == 'grad_uniform_convex':
            return \
                self.alpha * g_k_sqr_norm ** (0.5 * self. p / (self.p - 1)) <= \
                self.tolerance
        elif self.criterion == 'grad_norm_bound':
            return g_k_sqr_norm <= self.c * g_k_sqr_bound
        elif self.criterion == 'grad_norm_lambda_bound':
            return g_k_sqr_norm <= self.c * self.lambda_bound(T)
        elif self.criterion == 'none':
            return False
        else:
            raise ValueError(
                'Unknown stopping criterion {}'.format(self.criterion))

def get_tolerance(tolerance_options=None):
    """
    Helper to construct Tolerance.
    """
    if tolerance_options:
        if type(tolerance_options) is Tolerance:
            return tolerance_options
        else:
            return Tolerance.from_dict(tolerance_options)
    else:
        return Tolerance()

class ToleranceStrategy(object):
    """
    Choosing the inner tolerance.
    """
    def __init__(self, strategy='constant', **kwargs):
        self.strategy = strategy
        
        if strategy == 'constant':
            self.delta = kwargs.get('delta', 1e-8)
            self.label = kwargs.get('label', ('%f' % self.delta))
        elif strategy == 'power':
            self.alpha = kwargs.get('alpha', 2)
            self.c = kwargs.get('c', 1.0)
            self.label = kwargs.get('label', ('1/k^%d' % self.alpha))
        elif strategy == 'adaptive':
            self.alpha = kwargs.get('alpha', 1)
            self.c = kwargs.get('c', 1.0)
            self.label = kwargs.get('label', 'adaptive')
        else:
            raise ValueError(
                'Unknown tolerance strategy {}'.format(strategy))

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__
        
    def get_tolerance(self, k, f_prev, f_cur):
        if self.strategy == 'constant':
            return self.delta
        elif self.strategy == 'power':
            return self.c / (k + 1) ** self.alpha
        elif self.strategy == 'adaptive':
            if f_prev is None or f_prev <= f_cur + 1e-9:
                return 1e-9
            else:
                return self.c * (f_prev - f_cur) ** self.alpha
        else:
            raise ValueError(
                'Unknown tolerance strategy {}'.format(strategy))

def get_tolerance_strategy(tolerance_strategy_options=None):
    """
    Helper to construct ToleranceStrategy.
    """
    if tolerance_strategy_options:
        if type(tolerance_strategy_options) is ToleranceStrategy:
            return tolerance_strategy_options
        else:
            return ToleranceStrategy.from_dict(tolerance_strategy_options)
    else:
        return ToleranceStrategy()
