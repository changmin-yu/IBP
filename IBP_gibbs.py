from typing import Dict, Any, Optional

import numpy as np

import random

from tqdm import trange

from utils.general_utils import (
    sampling_bernoulli, 
    expand_Z, 
    log_factorial, 
    log_Poisson_prob, 
    bounded_random_walk, 
)
from utils.data import binary_LDS


class IndianBuffetProcessGibbs:
    """
    Gibbs sampling for posterior inference in Indian Buffet Process.
    We assume fully Bayesian treatment with priors on the hyperparameters.
    """
    def __init__(
        self, 
        X: np.ndarray, 
        num_iters: int = 1000, 
        prior_params: Dict[str, Any] = {}, 
    ):
        self.X = X
        self.N, self.D = X.shape
        
        self.num_iters = num_iters
        
        self.initialise(prior_params)
        
    def initialise(self, prior_params: Dict[str, Any] = {}):
        # alpha
        self.alpha = prior_params["init_alpha"]
        self.prior_alpha_a = prior_params["prior_alpha_a"]
        self.prior_alpha_b = prior_params["prior_alpha_b"]
        
        # sigma_x
        self.sigma_x = prior_params["init_sigma_x"]
        self.prior_sigma_x_a = prior_params["prior_sigma_x_a"]
        self.prior_sigma_x_b = prior_params["prior_sigma_x_b"]
        
        # sigma_A
        self.sigma_A = prior_params["init_sigma_A"]
        self.prior_sigma_A_a = prior_params["prior_sigma_A_a"]
        self.prior_sigma_A_b = prior_params["prior_sigma_A_b"]
        
        self.initialise_Z()
    
    def initialise_Z(self):
        Z = np.zeros((self.N, 0))
        K = 0
        
        for n in range(self.N):
            for k in range(K):
                mk = np.sum(Z[:, k])
                Z[n, k] = sampling_bernoulli(mk/(n+1)) # TODO: 1 / (n+1)?
                # Z[n, k] = sampling_bernoulli(1 / (n + 1))
            K_new = np.random.poisson(self.alpha / (n+1))
            K += K_new
            
            Z = expand_Z(Z, n, K_new)
        
        self.Z = Z
        self.K = K
        
        assert Z.shape[-1] == K
    
    def gibbs_sampling(self):
        history = {
            "Z": [], 
            "K": np.zeros((self.num_iters, )), 
            "sigma_x": np.zeros((self.num_iters, )), 
            "sigma_A": np.zeros((self.num_iters, )), 
            "alpha": np.zeros((self.num_iters, )), 
            "log_p_X_Z": np.zeros((self.num_iters, )), 
        }
        
        with trange(self.num_iters, dynamic_ncols=True) as pbar:
            for i in pbar:
                self.step()
                
                log_p_X_Z = self.log_p_X_Z()
                
                history["Z"].append(self.Z)
                history["K"][i] = self.K
                history["sigma_x"][i] = self.sigma_x
                history["sigma_A"][i] = self.sigma_A
                history["alpha"][i] = self.alpha
                history["log_p_X_Z"][i] = log_p_X_Z
                
                pbar.set_description(f"Current K = {self.K}")
        
        return history
        
    def step(self):
        for n in random.sample(range(self.N), self.N):
            self.sample_Z(n)
            self.sample_K(n)
            
        self.sample_alpha()
        self.sample_sigma_x()
        self.sample_sigma_A()
        
    def log_conditional_X_given_Z(
        self, 
        Z: Optional[np.ndarray] = None, 
        sigma_x: Optional[float] = None, 
        sigma_A: Optional[float] = None, 
    ):
        if Z is None:
            Z = self.Z
        if sigma_x is None:
            sigma_x = self.sigma_x
        if sigma_A is None:
            sigma_A = self.sigma_A
            
        K = Z.shape[-1]
        M = np.linalg.inv(Z.T @ Z + np.square(sigma_x / sigma_A) * np.eye(K))
        exponent = -1 / (2 * sigma_x ** 2) * np.trace(
            self.X.T @ (np.eye(self.N) - Z @ M @ Z.T) @ self.X
        )
        normalising_constant = -self.N * self.D * np.log(2 * np.pi) / 2  - \
            self.D * (self.N - K) * np.log(sigma_x) - \
                K * self.D * np.log(sigma_A) + \
                    self.D * np.log(np.linalg.det(M)) / 2
        
        return normalising_constant + exponent
    
    def sample_Z(self, n: int):
        for k in range(self.K):
            mk = np.sum(self.Z[:, k]) - self.Z[n, k]
            if mk == 0:
                self.Z[n, k] = 0
            else:
                Z0 = self.Z + 0
                Z1 = self.Z + 0
                Z0[n, k] = 0
                Z1[n, k] = 1
                log_ratio = self.log_conditional_X_given_Z(Z1) - self.log_conditional_X_given_Z(Z0) + np.log(mk) - np.log(self.N - mk)
                self.Z[n, k] = sampling_bernoulli(log_ratio, type="logdiff")
    
    def sample_K(self, n: int, log_threshold: float = -16.0):
        log_probs = np.array([])
        K_new = 0
        # lmd = self.alpha / (n + 1) # TODO: self.alpha / self.N?
        lmd = self.alpha / self.N
        
        while log_Poisson_prob(K_new, lmd) > log_threshold:
            log_probs = np.append(log_probs, log_Poisson_prob(K_new, lmd) + self.log_conditional_X_given_Z(expand_Z(self.Z, n, K_new)))
            K_new += 1
        
        if len(log_probs) > 0:
            # log_probs = np.array(log_probs)
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)
            
            K_posterior = np.argmax(np.random.multinomial(1, probs)) # np.random.choice(np.arange(K_new), size=(1, ), replace=False, p=probs)
        else:
            K_posterior = 0
            
        m = np.sum(self.Z, axis=0) - self.Z[n, :]
        self.Z = expand_Z(self.Z[:, m != 0], n, K_posterior)
        self.K = self.Z.shape[1]
        
    def sample_alpha(self):
        posterior_alpha_a = self.prior_alpha_a + self.K
        posterior_alpha_b = self.prior_alpha_b + 1. / np.sum(1. / np.arange(1, self.N + 1))
        
        self.alpha = np.random.gamma(posterior_alpha_a, posterior_alpha_b)
        
    def sample_sigma_x(self, epsilon: float = 0.01):
        new_sigma_x = bounded_random_walk(self.sigma_x, epsilon, boundary=[0, None])
        log_p = (self.prior_sigma_x_a - 1) * (np.log(new_sigma_x) - np.log(self.sigma_x)) - \
            self.prior_sigma_x_b * (1 / new_sigma_x - 1 / self.sigma_x) + \
                self.log_conditional_X_given_Z(sigma_x=new_sigma_x) - \
                    self.log_conditional_X_given_Z()
        
        # if sampling_bernoulli(min(0, log_p), type="logdiff"): # `TODO: log type?
        if sampling_bernoulli(min(0, log_p), type="log"):
            self.sigma_x = new_sigma_x
        
    def sample_sigma_A(self, epsilon: float = 0.01):
        new_sigma_A = bounded_random_walk(self.sigma_A, epsilon, boundary=[0, None])
        log_p = (self.prior_sigma_A_a - 1) * (np.log(new_sigma_A) - np.log(self.sigma_A)) - \
            self.prior_sigma_A_b * (1 / new_sigma_A - 1 / self.sigma_A) + \
                self.log_conditional_X_given_Z(sigma_A=new_sigma_A) - \
                    self.log_conditional_X_given_Z()
        
        # if sampling_bernoulli(min(0, log_p), type="log"): # `TODO: should we use logdiff type?
        if sampling_bernoulli(min(0, log_p), type="log"):
            self.sigma_A = new_sigma_A
    
    def posterior_mean_A(self):
        K = self.Z.shape[-1]
        return np.linalg.inv(self.Z.T @ self.Z + np.square(self.sigma_x / self.sigma_A) * np.eye(K)) @ self.Z.T @ self.X
    
    def log_p_Z(self):
        K = self.Z.shape[-1]
        log_p_Z = -self.alpha * np.sum(1. / np.arange(1, self.N + 1)) + K * np.log(self.alpha)
        
        K_curr = 0
        
        for n in range(self.N):
            if n == 0:
                if np.sum(self.Z[n] == 1) > 0:
                    K1 = np.where(self.Z[n] == 1)[0][-1]
                else:
                    K1 = 0
            else:
                if np.sum(self.Z[n] == 1) > 0:
                    K_new = np.where(self.Z[n] == 1)[0][-1]
                    K1 = K_new - K_curr
                else:
                    K1 = 0
            K_curr += K1
            if K1 > 0:
                log_p_Z -= np.log(K1)
        
        for k in range(K):
            mk = np.sum(self.Z[:, k])
            log_p_Z += log_factorial(self.N - mk) + log_factorial(mk - 1) - log_factorial(self.N)
        
        return log_p_Z
    
    def log_p_X_Z(self):
        return self.log_conditional_X_given_Z() + self.log_p_Z()


if __name__=="__main__":
    seed = 2
    X, weights = binary_LDS(num_samples=100, noise_scale=0.1, binary_prob=0.5, seed = 2)
    
    print(X.shape)
    
    prior_params = {
        "init_alpha": 1.0, 
        "prior_alpha_a": 1.0,
        "prior_alpha_b": 1.0,
        "init_sigma_x": 1.0, 
        "prior_sigma_x_a": 1.0, 
        "prior_sigma_x_b": 1.0, 
        "init_sigma_A": 1.0, 
        "prior_sigma_A_a": 1.0, 
        "prior_sigma_A_b": 1.0, 
    }

    ibp = IndianBuffetProcessGibbs(X, num_iters=1000, prior_params=prior_params)
    
    history = ibp.gibbs_sampling()
    
    pass