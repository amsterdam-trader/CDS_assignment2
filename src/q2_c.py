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
        Fit the spatial score model by maximizing the log-likelihood
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run all analyses for Question 2(c)."""
    import matplotlib.pyplot as plt

    output_dir = ROOT / "figures_q2"
    output_dir.mkdir(exist_ok=True)

    np.random.seed(42)

    T = 500
    n = 9
    k = 3

    W_raw = np.random.rand(n, n)
    np.fill_diagonal(W_raw, 0)
    W = W_raw / W_raw.sum(axis=1, keepdims=True)

    omega_true = 0.05
    A_true = 0.05
    B_true = 0.80
    beta_true = np.array([0.0, 1.5, -0.5])
    sigma2_true = 2.0
    nu_true = 5.0

    t_grid = np.arange(T)
    rho_true = 0.5 + 0.4 * np.cos(2 * np.pi * t_grid / 200)

    print("QUESTION 2(c): Time-Varying Spatial Score Model")
    print(f"\nSimulation Setup:")
    print(f"  T={T}, n={n}, k={k}")
    print(f"  True params: omega={omega_true}, A={A_true}, B={B_true}")
    print(f"  Student's t df: {nu_true}")

    print("\nSimulating data ...")
    y_data, X_data = simulate_spatial_data(
        T, n, W, rho_true, beta_true, sigma2_true, nu_true
    )
    print(f"  Data shape: y={y_data.shape}, X={X_data.shape}")

    print("Estimating spatial score model ...")
    model = SpatialScoreModel(max_rho=0.99)
    result = model.fit(y_data, X_data, W, verbose=True)

    print("\nEstimation Results:")
    print(f"  Convergence: {'Yes' if result.success else 'No'}")
    print(f"  Log-likelihood: {-result.fun:.2f}")
    print(f"\n  omega  = {model.params['omega']:.4f}  (true: {omega_true})")
    print(f"  A      = {model.params['A']:.4f}  (true: {A_true})")
    print(f"  B      = {model.params['B']:.4f}  (true: {B_true})")
    print(f"  beta   = {model.params['beta']}")
    print(f"  sigma2 = {model.params['sigma2']:.4f}  (true: {sigma2_true})")
    print(f"  nu     = {model.params['nu']:.4f}  (true: {nu_true})")

    rho_unc = model.get_unconditional_mean()
    print(f"\n  Unconditional mean of rho: {rho_unc:.4f}")

    n_params = len(result.x)
    print(f"  AIC: {compute_aic(-result.fun, n_params):.2f}")
    print(f"  BIC: {compute_bic(-result.fun, n_params, T * n):.2f}")

    # ── Figure 1: True vs Filtered rho ──
    print("\nCreating visualizations ...")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rho_true, 'k-', lw=2, label='True', alpha=0.7)
    ax.plot(model.filtered_rho, 'r--', lw=2, label='Filtered', alpha=0.7)
    ax.set_xlabel('Time'); ax.set_ylabel(r'$\rho(t)$')
    ax.set_title('Time-Varying Spatial Dependence: True vs Filtered')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "q2c_rho_comparison.png", dpi=150)
    plt.show(); plt.close()

    # ── Figure 2: Model diagnostics (4-panel) ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(model.filtered_rho, 'b-', lw=1.5)
    axes[0, 0].axhline(rho_unc, color='r', ls='--',
                        label=f'Uncond. mean: {rho_unc:.3f}')
    axes[0, 0].set_title('Filtered rho'); axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(model.filtered_rho, bins=30, edgecolor='k', alpha=0.7)
    axes[0, 1].axvline(rho_unc, color='r', ls='--', lw=2)
    axes[0, 1].set_title('Distribution of rho')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for i in range(min(3, n)):
        axes[1, 0].plot(y_data[:, i], alpha=0.7, label=f'Unit {i+1}')
    axes[1, 0].set_title('Sample time series'); axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    residuals = np.zeros((T, n))
    for t in range(T):
        residuals[t] = (y_data[t]
                        - model.filtered_rho[t] * (W @ y_data[t])
                        - X_data[t] @ model.params['beta'])
    axes[1, 1].plot(residuals.mean(axis=1), 'g-', lw=1.5)
    axes[1, 1].axhline(0, color='k', ls='--', lw=1)
    axes[1, 1].set_title('Average residuals'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "q2c_model_diagnostics.png", dpi=150)
    plt.show(); plt.close()

    # ── Figure 3: Different rho patterns ──
    patterns = {
        'Constant': np.full(T, 0.9),
        'Sine':     0.5 + 0.4 * np.cos(2 * np.pi * t_grid / 200),
        'Step':     0.9 - 0.5 * (t_grid > T / 2).astype(float),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (name, pattern) in enumerate(patterns.items()):
        y_sim, X_sim = simulate_spatial_data(
            T, n, W, pattern, beta_true, sigma2_true, nu_true
        )
        m = SpatialScoreModel(max_rho=0.99)
        m.fit(y_sim, X_sim, W, verbose=False)
        axes[idx].plot(pattern, 'k-', lw=2, label='True', alpha=0.7)
        axes[idx].plot(m.filtered_rho, 'r--', lw=2, label='Filtered', alpha=0.7)
        axes[idx].set_title(name); axes[idx].legend(fontsize=9)
        axes[idx].set_ylim([0, 1]); axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "q2c_pattern_comparison.png", dpi=150)
    plt.show(); plt.close()

    print("\nQUESTION 2(c) COMPLETE")


if __name__ == "__main__":
    main()