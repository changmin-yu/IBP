from typing import Optional
import numpy as np


def binary_LDS(
    basis: Optional[np.ndarray] = None, 
    num_samples: int = 100, 
    noise_scale: float = 1.0, 
    binary_prob: float = 0.5, 
    seed: Optional[int] = None, 
):
    if seed is not None:
        np.random.seed(seed)
        
    if basis is None:
        basis = np.array([
            [[1, 0, 0, 0, 0, 0], 
             [1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0], 
             [1, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0]], 
            [[0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0], 
             [1, 0, 0, 0, 0, 0], 
             [1, 1, 1, 0, 0, 0]], 
            [[0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 1, 1], 
             [0, 0, 0, 0, 1, 1]], 
            [[0, 0, 0, 0, 1, 1], 
             [0, 0, 0, 0, 0, 1], 
             [0, 0, 0, 0, 0, 1], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0]], 
            [[0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 1, 1, 1, 0, 0], 
             [0, 0, 1, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0], 
             [0, 0, 0, 0, 0, 0]]
        ])
    
    num_features = len(basis)
    if len(basis.shape) == 3:
        basis = basis.reshape(num_features, -1)
    
    D = basis.shape[-1]
    
    weights = np.random.binomial(1, binary_prob, size=(num_samples, num_features))
    
    X = weights @ basis + np.random.normal(
        loc=0.0, scale=noise_scale, size=(num_samples, D)
    )
    
    return X, weights


if __name__=="__main__":
    X = binary_LDS(num_samples=100, noise_scale=0.1)