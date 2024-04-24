from typing import Optional

import numpy as np
from scipy.special import digamma, gammaln

from tqdm import trange

from utils.general_utils import sigmoid
from utils.data import binary_LDS

EPS = 1e-20


class IndianBuffetProcessInfiniteVI:
    def __init__(
        self, 
        K: int, # truncation of stick-breaking
        X: np.ndarray, 
        alpha: float, 
        sigmasq_A: float, 
        sigmasq_n: float, 
        threshold: float = 1e-3, 
        max_iter: int = 100, 
        verbose: bool = False, 
    ):
        self.K = K
        self.X = X
        
        self.N, self.D = X.shape
        
        self.alpha = alpha
        self.sigmasq_A = sigmasq_A
        self.sigmasq_n = sigmasq_n
        
        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.initialise()
        
    def initialise(self):
        # variational parameters for Z
        self.rho = np.random.random((self.N, self.K))
        
        # variational parameters for A
        self.m = np.zeros((self.K, self.D))
        self.sigmasq_A_variational = np.ones((self.K, ))
        
        # variational parameters for the sticking-breaking weight, v
        self.a = np.ones((self.K, ))
        self.b = np.ones((self.K, ))
        
    def fit(self):
        free_energy_history = []
        converge = False
        
        with trange(self.max_iter, dynamic_ncols=True) as pbar:
            for i in pbar:
                self.inference_step()
                
                free_energy = self.free_energy()
                free_energy_history.append(free_energy)
                
                if self.verbose:
                    self.print_log(i, free_energy)
                
                pbar.set_description(f"free_energy={free_energy:.2f}")
                
                # if len(free_energy_history) > 1:
                #     if np.abs(100 * (free_energy_history[-1] - free_energy_history[-2]) / np.abs(free_energy_history[-2])) < self.threshold:
                #         converge = True
                #         break
        
        return free_energy_history, converge
    
    def inference_step(self):
        self.update_A()
        self.update_Z()
        self.update_v()
        
    def free_energy(self):
        log_p_X_cond_Z_A = self.exp_log_p_X_cond_Z_A()
        exp_log_p_V = np.sum(
            np.log(self.alpha) + (self.alpha - 1) * (digamma(self.a) - digamma(self.a + self.b))
        )
        exp_log_p_A = -self.D * self.K / 2 * np.log(2 * np.pi * self.sigmasq_A) 
        for k in range(self.K):
            exp_log_p_A -= 1 / (2 * self.sigmasq_A) * np.sum(np.square(self.m[k]) + self.sigmasq_A_variational[k])
        exp_log_p_Z = 0.0
        for k in range(self.K):
            exp_log_p_Z += np.sum(
                self.rho[:, k] * np.sum(digamma(self.a[:k]) - digamma(self.a[:k] + self.b[:k])) + # TODO: verify if it should be digamma(self.b[:k])
                (1 - self.rho[:, k]) * self.exp_log_1_minus_prod_v(k)
            )
        
        neg_exp_log_q_V = np.sum(
            gammaln(self.a) + gammaln(self.b) - gammaln(self.a + self.b) - 
            (self.a - 1) * digamma(self.a) - 
            (self.b - 1) * digamma(self.b) + 
            (self.a + self.b - 2) * digamma(self.a + self.b)
        )
        neg_exp_log_q_A = np.sum(self.D / 2 * np.log(2 * np.pi * np.exp(1)) + self.D / 2 * np.log(self.sigmasq_A_variational))
        neg_exp_log_q_Z = np.sum(
            -self.rho * np.log(self.rho + EPS) - (1 - self.rho) * np.log(1 - self.rho + EPS)
        )
        
        return log_p_X_cond_Z_A + exp_log_p_V + exp_log_p_A + exp_log_p_Z + neg_exp_log_q_V + neg_exp_log_q_A + neg_exp_log_q_Z
    
    def update_Z(self):
        for n in range(self.N):
            for k in range(self.K):
                logit = (
                    -1 / (2 * self.sigmasq_n) * np.sum(self.sigmasq_A_variational[k] + np.square(self.m[k, :])) + 
                    1 / self.sigmasq_n * self.m[k].dot(self.X[n] - (np.dot(self.rho[n], self.m) - self.rho[n, k] * self.m[k])) + 
                    np.sum(digamma(self.a[:k]) - digamma(self.a[:k] + self.b[:k])) - 
                    self.exp_log_1_minus_prod_v(k)
                )
                self.rho[n, k] = sigmoid(logit)
    
    def update_v(self):
        for k in range(self.K):
            self.a[k] = self.alpha + np.sum(self.rho[:, k])
            self.b[k] = 1.
            for m in range(k, self.K):
                multinomial_probs = self.variational_multinomial(m+1)
                if m > k:
                    self.a[k] += (self.N - np.sum(self.rho[:, m])) * np.sum(multinomial_probs[(k+1):])
                self.b[k] += (self.N - np.sum(self.rho[:, m])) * multinomial_probs[k]
            
    def update_A(self):
        for k in range(self.K):
            self.sigmasq_A_variational[k] = 1 / (1 / (self.sigmasq_n) * np.sum(self.rho[:, k]) + 1 / (self.sigmasq_A)) # TODO: do we need softplus?
            self.m[k] = self.sigmasq_A_variational[k] / self.sigmasq_n * np.sum(self.rho[:, [k]] * (self.X - (np.matmul(self.rho, self.m) - np.outer(self.rho[:, k], self.m[k]))), axis=0)
            
    def exp_log_p_X_cond_Z_A(self):
        constant = -self.D * self.N / 2 * np.log(2 * np.pi * self.sigmasq_n)
        
        XTX = np.square(self.X).sum()
        XTZA = np.sum(-2 * self.rho[..., None] * self.m[None] * self.X[:, None, :])
        cross_terms = 2 * np.triu((self.m @ self.m.T) * (self.rho.T @ self.rho)).sum()
        diagonal_terms = 0.0
        for k in range(self.K):
            diagonal_terms += np.sum(self.rho[:, k] * np.sum(np.square(self.m[k]) + self.sigmasq_A_variational[k]))
        
        return constant + (-1 / (self.sigmasq_n) * (XTX + XTZA + cross_terms + diagonal_terms))
    
    def exp_log_1_minus_prod_v(self, K: Optional[int] = None):
        if K is None:
            K = self.K
        multinomial_probs = self.variational_multinomial(K)
        
        exp = 0.0
        for k in range(K):
            exp += multinomial_probs[k] * digamma(self.b[k])
            for m in range(k):
                if m < (k - 1):
                    exp += np.sum(multinomial_probs[(m+1):]) * digamma(self.a[m])
                exp -= np.sum(multinomial_probs[m:]) * digamma(self.a[m] + self.b[m])
        exp -= np.sum(multinomial_probs * np.log(multinomial_probs))
        
        return exp
    
    def variational_multinomial(self, K: int):
        q_y = np.zeros((K, ))
        for k in range(K):
            q_y[k] = np.exp(
                digamma(self.b[k]) + 
                np.sum(digamma(self.a[:(k-1)])) - 
                np.sum(digamma(self.a[:k] + self.b[:k]))
            )
        q_y /= np.sum(q_y)
        
        return q_y
    
    def print_log(self, it: int, free_energy: float):
        factors = np.where(self.rho > 0.5)[1]
        print(f"Iteration {it} | {len(np.unique(factors))} factors: {np.unique(factors)} | free energy: {free_energy:.2f}")


if __name__=="__main__":
    seed = 2
    X, weights = binary_LDS(num_samples=100, noise_scale=0.1, binary_prob=0.5, seed = 2)
    
    print(X.shape)
    
    K = 10
    alpha = 1.0
    sigmasq_A = 0.1
    sigmasq_n = 0.1
    
    IBP_VI_infinite = IndianBuffetProcessInfiniteVI(
        X=X, 
        K=K, 
        alpha=alpha, 
        sigmasq_A=sigmasq_A, 
        sigmasq_n=sigmasq_n, 
        verbose=False, 
        threshold=1e-10, 
    )
    
    IBP_VI_infinite.fit()