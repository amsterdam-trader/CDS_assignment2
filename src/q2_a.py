# -*- coding: utf-8 -*-

# Question 2a
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for scripts and servers)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# set the path
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)

data = pd.read_csv(ROOT / "data" / "iowa_yield_05_10.csv")

# Project paths
ROOT = Path(__file__).resolve().parents[1]

# Constants
RHO_MAX = 0.95   # link function bound (slides p.55)
N_COUNTRIES = 8
K_REGRESSORS = 6  # [1, Vstoxx, Euribor-EONIA, Euribor, stock, spread]


# Data loading

def load_data(cds_path, weights_path):
    """
    Load CDS spread data and spatial weight matrix from Excel files.

    Returns

    Y : ndarray (T, n)
        CDS spread observations.
    Xt_all : ndarray (T, n, k)
        Regressor matrices for each t.
    W : ndarray (n, n)
        Row-normalised spatial weight matrix.
    WY_all : ndarray (T, n)
        Pre-computed W @ y_t for each t.
    col_names : list
        Column names from the CDS data file.
    """
    cds_raw     = pd.read_excel(cds_path, header=None)
    weights_raw = pd.read_excel(weights_path, header=None)

    col_names = cds_raw.iloc[0].tolist()
    data      = cds_raw.iloc[1:].astype(float).reset_index(drop=True)
    data.columns = col_names

    n = N_COUNTRIES
    T = len(data)

    Y         = data[col_names[:n]].values            # (T, n)
    X_common  = data[col_names[8:11]].values          # (T, 3): Vstoxx, Euribor-EONIA, Euribor
    X_stocks  = data[col_names[11:11+n]].values       # (T, n)
    X_spreads = data[col_names[11+n:11+2*n]].values   # (T, n)

    # Row-normalise the raw spatial weight matrix
    W_raw = weights_raw.iloc[1:, 1:].astype(float).values
    W     = W_raw / W_raw.sum(axis=1, keepdims=True)

    # Build X_t = [1_n | Vstoxx*1_n | Euribor-EONIA*1_n | Euribor*1_n | stock_t | spread_t]
    Xt_all = np.array([
        np.column_stack([
            np.ones(n),
            np.tile(X_common[t], (n, 1)),
            X_stocks[t].reshape(-1, 1),
            X_spreads[t].reshape(-1, 1),
        ])
        for t in range(T)
    ])  # (T, n, K_REGRESSORS)

    WY_all = (W @ Y.T).T   # (T, n)

    return Y, Xt_all, W, WY_all, col_names


# Link function

def link(f):
    """h(f) = 0.95 * tanh(f),  maps R -> (-0.95, 0.95)."""
    return RHO_MAX * np.tanh(f)

def link_dot(f):
    """First derivative h'(f) = 0.95 * (1 - tanh^2(f))."""
    return RHO_MAX * (1.0 - np.tanh(f) ** 2)

def link_ddot(f):
    """Second derivative h''(f) = -2 * 0.95 * tanh(f) * (1 - tanh^2(f))."""
    t = np.tanh(f)
    return -2.0 * RHO_MAX * t * (1.0 - t ** 2)


# Score-driven filter

def run_filter(omega, beta, alpha, log_sigma, beta_reg, Y, Xt_all, W, WY_all):
    """
    Run the Gaussian score-driven filter for one set of parameters.
    """
    T, n = Y.shape
    sigma2  = np.exp(2.0 * log_sigma)
    I_n     = np.eye(n)

    f   = np.empty(T)
    rho = np.empty(T)

    f[0]   = omega
    rho[0] = link(omega)

    total_ll = 0.0

    for t in range(T):
        S_t              = I_n - rho[t] * W
        sign, logdet_S   = np.linalg.slogdet(S_t)
        if sign <= 0:
            return -1e10, f, rho

        e_t  = S_t @ Y[t] - Xt_all[t] @ beta_reg
        ll_t = (logdet_S
                - (n / 2.0) * np.log(2.0 * np.pi * sigma2)
                - (e_t @ e_t) / (2.0 * sigma2))

        if not np.isfinite(ll_t):
            return -1e10, f, rho
        total_ll += ll_t

        if t < T - 1:
            S_inv     = np.linalg.solve(S_t, I_n)
            d_rho     = (WY_all[t] @ e_t) / sigma2 - np.trace(S_inv @ W)
            score_t   = d_rho * link_dot(f[t])
            f[t + 1]  = omega * (1.0 - beta) + beta * f[t] + alpha * score_t
            rho[t + 1] = link(f[t + 1])

    return total_ll, f, rho


def _neg_loglik(params, Y, Xt_all, W, WY_all):
    """Wrapper returning the negative log-likelihood for scipy.optimize."""
    ll, _, _ = run_filter(params[0], params[1], params[2], params[3],
                          params[4:], Y, Xt_all, W, WY_all)
    return -ll


# Estimation

def estimate_gaussian_model(Y, Xt_all, W, WY_all, bounds, n_starts=6):
    """
    Estimate the Gaussian score-driven model by MLE.

    Uses multiple starting points and returns the best result.

    Returns:
    params_opt : ndarray
        Estimated parameters [omega, beta, alpha, log_sigma, beta_reg].
    ll_opt : float
        Maximised log-likelihood.
    results_list : list
        List of (ll, params) for every starting point (for diagnostics).
    """
    k = K_REGRESSORS
    starts = [
        [0.5, 0.95, 0.05, np.log(3.0)] + [0.0] * k,
        [1.0, 0.90, 0.10, np.log(2.0)] + [0.0] * k,
        [0.5, 0.98, 0.02, np.log(5.0)] + [0.0] * k,
        [1.5, 0.95, 0.05, np.log(3.0)] + [0.0] * k,
        [0.5, 0.85, 0.20, np.log(3.0)] + [0.0] * k,
        [1.0, 0.97, 0.01, np.log(4.0)] + [0.0] * k,
    ]

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    best_params, best_ll = None, -np.inf
    results_list = []

    for x0 in starts[:n_starts]:
        x0  = np.clip(x0, lo, hi)
        res = minimize(
            _neg_loglik, x0,
            args=(Y, Xt_all, W, WY_all),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7},
        )
        ll = -res.fun
        results_list.append((ll, res.x))
        if ll > best_ll:
            best_ll     = ll
            best_params = res.x.copy()

    return best_params, best_ll, results_list


# QLR test

def _neg_loglik_restricted(params, Y, Xt_all, W, WY_all):
    """
    Negative log-likelihood for the restricted model (alpha = 0).

    When alpha = 0, f_t = omega for all t, so rho_t = h(omega) is constant.
    params = [omega, beta (ignored), log_sigma, beta_reg (k,)]
    """
    omega, log_sigma = params[0], params[2]
    beta_reg         = params[3:]

    rho_c  = link(omega)
    sigma2 = np.exp(2.0 * log_sigma)
    n      = Y.shape[1]
    I_n    = np.eye(n)
    S      = I_n - rho_c * W

    sign, logdet_S = np.linalg.slogdet(S)
    if sign <= 0:
        return np.inf

    total_ll = 0.0
    for t in range(len(Y)):
        e_t  = S @ Y[t] - Xt_all[t] @ beta_reg
        ll_t = (logdet_S
                - (n / 2.0) * np.log(2.0 * np.pi * sigma2)
                - (e_t @ e_t) / (2.0 * sigma2))
        if not np.isfinite(ll_t):
            return np.inf
        total_ll += ll_t

    return -total_ll


def compute_kappa_hat(omega_r, log_sig_r, beta_reg_r, Y, Xt_all, W, WY_all):
    """
    Compute the scaling factor kappa_hat for the scaled QLR test (slides p.51-53).

    kappa_hat = [-T^{-1} sum_t nabla_ff_t]^{-1} * [T^{-1} sum_t (nabla_f_t)^2]

    All quantities evaluated at the restricted estimates under H0.

    Parameters:
    omega_r, log_sig_r : float
        Restricted estimates of omega and log(sigma).
    beta_reg_r : ndarray (k,)
        Restricted regression coefficient estimates.
    Y, Xt_all, W, WY_all : arrays
        Pre-loaded data.

    Returns:
    kappa_hat : float
    """
    T, n   = Y.shape
    rho_rv = link(omega_r)
    sigma2 = np.exp(2.0 * log_sig_r)
    I_n    = np.eye(n)

    S       = I_n - rho_rv * W
    S_inv   = np.linalg.inv(S)
    SinvW   = S_inv @ W
    tr_1    = np.trace(SinvW)           # tr(S^{-1} W)
    tr_2    = np.trace(SinvW @ SinvW)   # tr((S^{-1} W)^2)
    h_dot   = link_dot(omega_r)
    h_ddot  = link_ddot(omega_r)

    sum_sq   = 0.0
    sum_hess = 0.0

    for t in range(T):
        e_t   = S @ Y[t] - Xt_all[t] @ beta_reg_r
        d_rho = (WY_all[t] @ e_t) / sigma2 - tr_1

        # nabla_f_t  = d_rho * h'(omega_r)
        score_t = d_rho * h_dot

        # nabla_ff_t = d_rho * h''(omega_r)
        #            - [sigma^{-2} y_t'W'W y_t + tr((S^{-1}W)^2)] * h'(omega_r)^2
        hess_t = (d_rho * h_ddot
                  - ((Y[t] @ (W.T @ WY_all[t])) / sigma2 + tr_2) * h_dot ** 2)

        sum_sq   += score_t ** 2
        sum_hess += hess_t

    kappa_hat = (-sum_hess / T) ** (-1) * (sum_sq / T)
    return kappa_hat


def qlr_test(ll_unrestricted, params_opt, Y, Xt_all, W, WY_all, bounds_restricted,
             beta_U=0.99, alpha_L=0):
    """
    QLR test for absence of score-driven dynamics (H0: alpha = 0).

    Computes the raw QLRT and the scaled QLR_tilde statistic, and reports
    critical values from Table 1 of the assignment.

    Parameters:
    ll_unrestricted : float
        Maximised log-likelihood of the unrestricted model.
    params_opt : ndarray
        Unrestricted parameter estimates (used as starting point).
    Y, Xt_all, W, WY_all : arrays
        Pre-loaded data.
    bounds_restricted : list
        Bounds for restricted optimisation [omega, beta, log_sigma, beta_reg].
    beta_U : float
        Upper bound of B (used to select the correct row in Table 1).
    alpha_L : float
        Lower bound of Theta_alpha (0 or negative).

    Returns:
    dict with keys:
        ll_restricted, QLRT, kappa_hat, QLR_tilde,
        omega_r, rho_const, sigma_r, beta_reg_r,
        cv_10, cv_05, cv_01, conclusion
    """
    omega_m, beta_m, log_sig_m = params_opt[0], params_opt[1], params_opt[3]
    beta_reg_m = params_opt[4:]
    k = len(beta_reg_m)

    # --- Restricted estimation (alpha = 0) ---
    x0_candidates = [
        np.array([omega_m, beta_m, log_sig_m] + list(beta_reg_m)),
        np.array([0.5,     0.90,   np.log(3.0)] + [0.0] * k),
        np.array([1.0,     0.95,   np.log(2.0)] + [0.0] * k),
    ]
    lo_r = [b[0] for b in bounds_restricted]
    hi_r = [b[1] for b in bounds_restricted]

    best_ll_r, best_params_r = -np.inf, None
    for x0 in x0_candidates:
        x0  = np.clip(x0, lo_r, hi_r)
        res = minimize(
            _neg_loglik_restricted, x0,
            args=(Y, Xt_all, W, WY_all),
            method='L-BFGS-B',
            bounds=bounds_restricted,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7},
        )
        if -res.fun > best_ll_r:
            best_ll_r    = -res.fun
            best_params_r = res.x.copy()

    ll_restricted = best_ll_r
    omega_r, log_sig_r = best_params_r[0], best_params_r[2]
    beta_reg_r = best_params_r[3:]

    # Raw and scaled QLR statistics
    QLRT      = 2.0 * (ll_unrestricted - ll_restricted)
    kappa_hat = compute_kappa_hat(omega_r, log_sig_r, beta_reg_r,
                                  Y, Xt_all, W, WY_all)
    QLR_tilde = QLRT / kappa_hat

    # Critical values (Table 1, assignment)
    # Specification: B = [0, beta_U], Theta_alpha = [alpha_L, alpha_U]
    # With beta_U = 0.99, alpha_L = 0:
    #   Closest row is beta_L=0, beta_U=0.990 => 10%: 3.266, 5%: 4.613, 1%: 7.696
    #
    # Note: H0 is rejected if QLR_tilde >= critical value (slides p.56).
    cv_10, cv_05, cv_01 = 3.266, 4.613, 7.696

    if QLR_tilde >= cv_01:
        conclusion = "Reject H0 at 1% level: strong evidence of time-varying rho_t."
    elif QLR_tilde >= cv_05:
        conclusion = "Reject H0 at 5% level."
    elif QLR_tilde >= cv_10:
        conclusion = "Reject H0 at 10% level."
    else:
        conclusion = "Fail to reject H0: no evidence of time variation."

    return {
        'll_restricted' : ll_restricted,
        'QLRT'          : QLRT,
        'kappa_hat'     : kappa_hat,
        'QLR_tilde'     : QLR_tilde,
        'omega_r'       : omega_r,
        'rho_const'     : link(omega_r),
        'sigma_r'       : np.exp(log_sig_r),
        'beta_reg_r'    : beta_reg_r,
        'cv_10'         : cv_10,
        'cv_05'         : cv_05,
        'cv_01'         : cv_01,
        'conclusion'    : conclusion,
    }


# Plotting

def plot_rho_path(rho_main, rho_const, T, save_path='figures_q2/rho_path.png'):
    """
    Replicate the figure on slide p.56: time-varying rho_t vs. time-constant rho.

    Parameters
    rho_main : ndarray (T,)
        Estimated time-varying spillover path.
    rho_const : float
        Time-constant rho estimate (from restricted model).
    T : int
        Sample size.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rho_main, color='navy', linewidth=0.8, label=r'$\hat{\rho}_t$ (time-varying)')
    ax.axhline(y=rho_const, color='red', linestyle='--', linewidth=1.5,
               label=r'time-constant $\hat{\rho}$')
    ax.set_xlim(0, T)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\hat{\rho}_t$', fontsize=12)
    ax.set_title('Gaussian score-driven spatial spillover dynamics', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_rho_comparison(rho_paths, labels, rho_const, save_path='figures_q2/rho_comparison.png'):
    """
    Plot rho_t paths side-by-side for multiple specifications.

    Parameters:
    rho_paths : list of ndarray (T,)
    labels : list of str
    rho_const : float
        Time-constant rho (from restricted model, shown as red dashed line).
    """
    n_specs = len(rho_paths)
    fig, axes = plt.subplots(1, n_specs, figsize=(5 * n_specs, 4), sharey=True)
    if n_specs == 1:
        axes = [axes]

    for ax, rho_path, label in zip(axes, rho_paths, labels):
        ax.plot(rho_path, color='navy', linewidth=0.7)
        ax.axhline(y=rho_const, color='red', linestyle='--', linewidth=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Time')
        ax.grid(True, linestyle=':', alpha=0.5)

    axes[0].set_ylabel(r'$\hat{\rho}_t$')
    fig.suptitle('Comparison of alternative parameter space specifications', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


# Summary printing

def print_estimates(label, params, ll, rho_path):
    """Print a compact summary of one model's estimates."""
    omega, beta, alpha, log_sig = params[:4]
    beta_reg = params[4:]
    print(f"\n{label}")
    print(f"  omega   = {omega:.6f}")
    print(f"  beta    = {beta:.6f}   (B upper bound)")
    print(f"  alpha   = {alpha:.6f}")
    print(f"  sigma   = {np.exp(log_sig):.6f}")
    print(f"  beta_reg = [{', '.join(f'{b:.4f}' for b in beta_reg)}]")
    print(f"  Log-likelihood = {ll:.4f}")
    print(f"  rho_t range    = [{rho_path.min():.4f}, {rho_path.max():.4f}]")


def print_qlr_results(qlr_results):
    """Print the QLR test output."""
    r = qlr_results
    print("\n" + "=" * 60)
    print("QLR TEST FOR TIME VARIATION  (H0: alpha = 0)")
    print("=" * 60)
    print(f"  Restricted model: rho_const = {r['rho_const']:.6f}  "
          f"(omega_r = {r['omega_r']:.6f}, sigma_r = {r['sigma_r']:.6f})")
    print(f"  LL (restricted)   = {r['ll_restricted']:.4f}")
    print(f"  QLRT   (raw)      = {r['QLRT']:.4f}")
    print(f"  kappa_hat         = {r['kappa_hat']:.6f}")
    print(f"  QLR_tilde         = {r['QLR_tilde']:.4f}")
    print(f"  Critical values   (Table 1, B=[0,0.99], Theta_alpha=[0,1], alpha_L=0):")
    print(f"    10%: {r['cv_10']},  5%: {r['cv_05']},  1%: {r['cv_01']}")
    print(f"  Conclusion: {r['conclusion']}")