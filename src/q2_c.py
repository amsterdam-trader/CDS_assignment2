# -*- coding: utf-8 -*-

# Question 2c
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import optimize
from scipy.special import digamma, gammaln
from scipy.linalg import inv, det
import warnings
warnings.filterwarnings('ignore')

# set the path
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)

data = pd.read_excel(ROOT / "data" / "cds_spatialweights.xlsx")

# Project paths
ROOT = Path(__file__).resolve().parents[1]


class SpatialScoreModel:
    """
    Time-varying Spatial Durbin Model with score-driven dynamics
    
    Model:
        y_t = ρ_t * W * y_t + X_t * β + e_t
        e_t ~ t_λ(0, Σ)
        
    Score dynamics:
        f_{t+1} = ω + A * s_t + B * f_t
        ρ_t = tanh(f_t)
    """
    
    def __init__(self, max_rho=0.99):
        """
        Parameters:
        max_rho : float
            Maximum absolute value for spatial correlation (default 0.99)
        """
        self.max_rho = max_rho
        self.params = None
        self.filtered_rho = None
        self.filtered_f = None
        
    def transform_rho(self, f):
        """Transform f to ρ ∈ (-max_rho, max_rho)"""
        return self.max_rho * np.tanh(f)
    
    def transform_rho_derivative(self, f):
        """Derivative of transformation function"""
        return self.max_rho * (1 - np.tanh(f)**2)
    
    def compute_Z_inv(self, rho, W):
        """Compute (I - ρW)"""
        n = W.shape[0]
        return np.eye(n) - rho * W
    
    def compute_score(self, y, X, W, f, beta, sigma2, nu):
        """
        Compute score for spatial dependence parameter
        
        Equation (12) from the paper
        """
        n = len(y)
        rho = self.transform_rho(f)
        h_dot = self.transform_rho_derivative(f)
        
        # Compute Z^{-1} = (I - ρW)
        Z_inv = self.compute_Z_inv(rho, W)
        
        # Compute residuals
        Wy = W @ y
        residual = y - rho * Wy - X @ beta
        
        # Student's t weight
        quad_form = (residual ** 2).sum() / sigma2
        w_tilde = (1 + 1/nu) / (1 + quad_form / nu)
        
        # Score components
        spatial_term = w_tilde * (Wy.T @ residual) / sigma2
        trace_term = np.trace(inv(Z_inv) @ W)
        
        score = (spatial_term - trace_term) * h_dot
        
        return score
    
    def log_likelihood_t(self, y, X, W, f, beta, sigma2, nu):
        """
        Compute log-likelihood for time t with Student's t errors
        
        Equation (9) from the paper
        """
        n = len(y)
        rho = self.transform_rho(f)
        
        # Compute Z^{-1} = (I - ρW)
        Z_inv = self.compute_Z_inv(rho, W)
        log_det_Z_inv = np.log(np.abs(det(Z_inv)))
        
        # Compute residuals
        residual = y - rho * (W @ y) - X @ beta
        
        # Student's t log-likelihood
        quad_form = (residual ** 2).sum() / sigma2
        
        ll = (log_det_Z_inv 
              + gammaln((nu + n) / 2) 
              - gammaln(nu / 2)
              - 0.5 * n * np.log(sigma2)
              - 0.5 * n * np.log(nu * np.pi)
              - ((nu + n) / 2) * np.log(1 + quad_form / nu))
        
        return ll
    
    def filter_rho(self, y_data, X_data, W, omega, A, B, beta, sigma2, nu, f_init=0.0):
        """
        Filter the time-varying spatial dependence parameter
        
        Parameters:
        y_data : array (T, n)
            Dependent variable panel data
        X_data : array (T, n, k)
            Regressors panel data
        W : array (n, n)
            Spatial weights matrix
        omega, A, B : float
            Score dynamics parameters
        beta : array (k,)
            Regression coefficients
        sigma2 : float
            Error variance
        nu : float
            Degrees of freedom for Student's t
        f_init : float
            Initial value for f_1
            
        Returns:
        f_filtered : array (T,)
            Filtered values of f_t
        rho_filtered : array (T,)
            Filtered values of ρ_t
        """
        T = y_data.shape[0]
        f_filtered = np.zeros(T)
        rho_filtered = np.zeros(T)
        
        f_t = f_init
        
        for t in range(T):
            y = y_data[t]
            X = X_data[t]
            
            # Store filtered value
            f_filtered[t] = f_t
            rho_filtered[t] = self.transform_rho(f_t)
            
            # Compute score
            score = self.compute_score(y, X, W, f_t, beta, sigma2, nu)
            
            # Update f for next period
            if t < T - 1:
                f_t = omega + A * score + B * f_t
        
        return f_filtered, rho_filtered
    
    def negative_log_likelihood(self, params, y_data, X_data, W, f_init=0.0):
        """
        Negative log-likelihood for optimization
        
        Parameters:
        params : array
            [omega, A, B, beta_0, ..., beta_k, log(sigma2), log(nu)]
        """
        T, n = y_data.shape
        k = X_data.shape[2]
        
        # Unpack parameters
        omega = params[0]
        A = params[1]
        B = params[2]
        beta = params[3:3+k]
        sigma2 = np.exp(params[3+k])
        nu = np.exp(params[3+k+1])
        
        # Ensure stability |B| < 1
        if np.abs(B) >= 0.999:
            return 1e10
        
        # Filter f_t and compute likelihood
        f_t = f_init
        total_ll = 0.0
        
        for t in range(T):
            y = y_data[t]
            X = X_data[t]
            
            # Compute log-likelihood at time t
            ll_t = self.log_likelihood_t(y, X, W, f_t, beta, sigma2, nu)
            total_ll += ll_t
            
            # Compute score and update f for next period
            score = self.compute_score(y, X, W, f_t, beta, sigma2, nu)
            f_t = omega + A * score + B * f_t
        
        return -total_ll
    
    def fit(self, y_data, X_data, W, initial_params=None, f_init=0.0, 
            method='L-BFGS-B', maxiter=1000, verbose=False):
        """
        Estimate model parameters via Maximum Likelihood
        
        Parameters:
        y_data : array (T, n)
            Dependent variable panel data
        X_data : array (T, n, k)
            Regressors panel data
        W : array (n, n)
            Spatial weights matrix
        initial_params : array, optional
            Initial parameter values
        f_init : float
            Initial value for f_1
        method : str
            Optimization method
        maxiter : int
            Maximum iterations
        verbose : bool
            Print optimization progress
            
        Returns:
        result : OptimizeResult
            Optimization result object
        """
        T, n = y_data.shape
        k = X_data.shape[2]
        
        # Initialize parameters if not provided
        if initial_params is None:
            omega = 0.01
            A = 0.01
            B = 0.98
            beta = np.zeros(k)
            log_sigma2 = np.log(1.0)
            log_nu = np.log(5.0)
            initial_params = np.array([omega, A, B] + beta.tolist() + [log_sigma2, log_nu])
        
        # Bounds for parameters
        bounds = (
            [(None, None)] * 3 +  # omega, A, B
            [(None, None)] * k +   # beta
            [(None, None)] +       # log(sigma2)
            [(None, None)]         # log(nu)
        )
        
        # Optimize
        if verbose:
            print("Starting optimization")
            options = {'disp': True, 'maxiter': maxiter}
        else:
            options = {'maxiter': maxiter}
        
        result = optimize.minimize(
            self.negative_log_likelihood,
            initial_params,
            args=(y_data, X_data, W, f_init),
            method=method,
            bounds=bounds,
            options=options
        )
        
        if verbose:
            print(f"Optimization completed. Success: {result.success}")
            print(f"Final log-likelihood: {-result.fun:.2f}")
        
        # Store estimated parameters
        self.params = {
            'omega': result.x[0],
            'A': result.x[1],
            'B': result.x[2],
            'beta': result.x[3:3+k],
            'sigma2': np.exp(result.x[3+k]),
            'nu': np.exp(result.x[3+k+1])
        }
        
        # Compute filtered rho values
        self.filtered_f, self.filtered_rho = self.filter_rho(
            y_data, X_data, W,
            self.params['omega'],
            self.params['A'],
            self.params['B'],
            self.params['beta'],
            self.params['sigma2'],
            self.params['nu'],
            f_init
        )
        
        return result
    
    def get_unconditional_mean(self):
        """Compute unconditional mean of ρ_t"""
        if self.params is None:
            raise ValueError("Model must be fitted first")
        
        omega = self.params['omega']
        B = self.params['B']
        
        f_mean = omega / (1 - B)
        rho_mean = self.transform_rho(f_mean)
        
        return rho_mean


def simulate_spatial_data(T, n, W, rho_process, beta, sigma2, nu, X_data=None):
    """
    Simulate data from spatial model
    
    Parameters:
    T : int
        Number of time periods
    n : int
        Number of cross-sectional units
    W : array (n, n)
        Spatial weights matrix
    rho_process : array (T,)
        Time-varying spatial correlation
    beta : array (k,)
        Regression coefficients
    sigma2 : float
        Error variance
    nu : float
        Degrees of freedom
    X_data : array (T, n, k), optional
        Regressors (if None, standard normal)
        
    Returns:
    y_data : array (T, n)
        Simulated dependent variable
    X_data : array (T, n, k)
        Regressors
    """
    k = len(beta)
    
    if X_data is None:
        X_data = np.random.randn(T, n, k)
    
    y_data = np.zeros((T, n))
    
    for t in range(T):
        rho = rho_process[t]
        X = X_data[t]
        
        # Generate Student's t errors
        chi2 = np.random.chisquare(nu, size=n)
        z = np.random.randn(n)
        e = np.sqrt(nu * sigma2 / chi2) * z
        
        # Solve (I - ρW)y = Xβ + e
        I_minus_rhoW = np.eye(n) - rho * W
        y_data[t] = inv(I_minus_rhoW) @ (X @ beta + e)
    
    return y_data, X_data


def compute_aic(log_likelihood, n_params):
    """Compute AIC"""
    return -2 * log_likelihood + 2 * n_params


def compute_bic(log_likelihood, n_params, n_obs):
    """Compute BIC"""
    return -2 * log_likelihood + n_params * np.log(n_obs)