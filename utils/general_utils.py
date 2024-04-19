from typing import Optional, Union, List, Any

import numpy as np


def sampling_bernoulli(p: float, type: Optional[str] = None):
    if type == "log":
        if p > 0:
            return 1
        elif p < -200:
            return 0
        p = np.exp(p)
    
    elif type == "logdiff":
        if p > 200:
            return 1
        elif p < -200:
            return 0
        p = np.exp(p) / (1 + np.exp(p))
    
    return np.random.random() < p


def expand_Z(Z: np.ndarray, n: int, K_new: int):
    """
    Add K_new columns to the right of Z, where Z_new[n, -K_new:] = 1
    """
    N, K = Z.shape
    Z_new = np.zeros((N, K + K_new), dtype=int)
    
    Z_new[:, :K] = Z
    Z_new[n, K:] = 1
    
    return Z_new


def log_factorial(n: Union[int, np.ndarray]):
    if n == 0:
        return 0
    return np.log(n) + log_factorial(n-1)


def log_Poisson_prob(K: int, lmd: float):
    return K * np.log(lmd) - lmd - log_factorial(K)


def bounded_random_walk(
    x: Union[float, np.ndarray], 
    eps: float, 
    boundary: List[Any] = [None, None]
):
    new_x = np.random.normal(x, eps)
    left, right = boundary
    
    if left is not None and new_x < left:
        return 2 * left - new_x
    if right is not None and new_x > right:
        return 2 * right - new_x
    
    return new_x
