# -*- coding: utf-8 -*-
"""
Question 2(a) + 2(c)
- 2(a): Gaussian score-driven homogeneous SDM for CDS data + QLR test
- 2(c): Student-t score-driven homogeneous SDM (Blasques et al. 2016) and
        comparison plot (Gaussian vs Student-t filtered rho)

This file is meant to REPLACE your current q2_a.py (or you can save it as q2_ac.py).
It keeps the same structure as your q2_a.py, but:
  - fixes beta_reg bounds to [-10, 10] (slides)
  - clarifies Euribor regressor is "Euribor change"
  - removes misleading "(B upper bound)" printing
  - adds Student-t likelihood + robust weight and estimates it on the REAL CDS data
  - produces a Figure-4 style comparison plot: Gaussian (dashed) vs Student-t (solid)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)

RHO_MAX = 0.95
N_COUNTRIES = 8
K_REGRESSORS = 6  # [1, Vstoxx, Euribor-EONIA, Euribor change, stock, spread]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data(cds_path, weights_path):
    """
    Load CDS spread data and spatial weights from Excel.

    Returns
    -------
    Y : (T, n) ndarray
    Xt_all : (T, n, k) ndarray
    W : (n, n) ndarray row-normalized
    WY_all : (T, n) ndarray precomputed W @ y_t
    col_names : list
    """
    cds_raw = pd.read_excel(cds_path, header=None)
    weights_raw = pd.read_excel(weights_path, header=None)

    col_names = cds_raw.iloc[0].tolist()
    data = cds_raw.iloc[1:].astype(float).reset_index(drop=True)
    data.columns = col_names

    n = N_COUNTRIES
    T = len(data)

    # Y: first 8 columns
    Y = data[col_names[:n]].values  # (T, n)

    # Common regressors columns 9-11 (0-indexed 8:11)
    # These are: Vstoxx changes, Euribor-EONIA, Euribor change
    X_common = data[col_names[8:11]].values  # (T,3)

    # Country-specific regressors
    X_stocks = data[col_names[11:11 + n]].values       # (T,n)
    X_spreads = data[col_names[11 + n:11 + 2 * n]].values  # (T,n)

    # Weights: drop header row/col then row-normalize
    W_raw = weights_raw.iloc[1:, 1:].astype(float).values
    row_sums = W_raw.sum(axis=1, keepdims=True)
    W = W_raw / row_sums

    # Build X_t = [1_n | common(3) | stock_t | spread_t]  => k=6
    Xt_all = np.array([
        np.column_stack([
            np.ones(n),
            np.tile(X_common[t], (n, 1)),
            X_stocks[t].reshape(-1, 1),
            X_spreads[t].reshape(-1, 1),
        ])
        for t in range(T)
    ])  # (T,n,k)

    WY_all = (W @ Y.T).T  # (T,n)

    return Y, Xt_all, W, WY_all, col_names


# ---------------------------------------------------------------------
# Link function (slides)
# ---------------------------------------------------------------------
def link(f):
    """h(f)=0.95*tanh(f)"""
    return RHO_MAX * np.tanh(f)


def link_dot(f):
    """h'(f)=0.95*(1-tanh^2(f))"""
    t = np.tanh(f)
    return RHO_MAX * (1.0 - t * t)


def link_ddot(f):
    """h''(f)=-2*0.95*tanh(f)*(1-tanh^2(f))"""
    t = np.tanh(f)
    return -2.0 * RHO_MAX * t * (1.0 - t * t)


# ---------------------------------------------------------------------
# Gaussian score-driven filter (Q2a)
# ---------------------------------------------------------------------
def run_filter_gaussian(omega, B, A, log_sigma, beta_reg, Y, Xt_all, W, WY_all):
    """
    Gaussian score-driven homogeneous SDM filter.

    Returns
    -------
    total_ll : float
    f : (T,) ndarray
    rho : (T,) ndarray
    """
    T, n = Y.shape
    sigma2 = np.exp(2.0 * log_sigma)
    I_n = np.eye(n)

    f = np.empty(T)
    rho = np.empty(T)
    f[0] = omega
    rho[0] = link(f[0])

    total_ll = 0.0

    for t in range(T):
        S = I_n - rho[t] * W
        sign, logdetS = np.linalg.slogdet(S)
        if sign <= 0:
            return -1e12, f, rho

        e_t = S @ Y[t] - Xt_all[t] @ beta_reg

        ll_t = (
            logdetS
            - (n / 2.0) * np.log(2.0 * np.pi * sigma2)
            - (e_t @ e_t) / (2.0 * sigma2)
        )
        if not np.isfinite(ll_t):
            return -1e12, f, rho
        total_ll += ll_t

        if t < T - 1:
            S_inv = np.linalg.solve(S, I_n)
            d_rho = (WY_all[t] @ e_t) / sigma2 - np.trace(S_inv @ W)
            score_t = d_rho * link_dot(f[t])

            # slides intercept form
            f[t + 1] = omega * (1.0 - B) + B * f[t] + A * score_t
            rho[t + 1] = link(f[t + 1])

    return total_ll, f, rho


def _neg_loglik_gaussian(params, Y, Xt_all, W, WY_all):
    ll, _, _ = run_filter_gaussian(params[0], params[1], params[2], params[3],
                                   params[4:], Y, Xt_all, W, WY_all)
    return -ll


def estimate_gaussian_model(Y, Xt_all, W, WY_all, bounds, n_starts=6):
    """
    MLE for Gaussian model using multiple starting points.
    params = [omega, B, A, log_sigma, beta_reg(k)]
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
        x0 = np.clip(x0, lo, hi)
        res = minimize(
            _neg_loglik_gaussian, x0,
            args=(Y, Xt_all, W, WY_all),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7},
        )
        ll = -res.fun
        results_list.append((ll, res.x))
        if ll > best_ll:
            best_ll = ll
            best_params = res.x.copy()

    return best_params, best_ll, results_list


# ---------------------------------------------------------------------
# Restricted model + QLR test (Q2a)
# ---------------------------------------------------------------------
def _neg_loglik_restricted(params, Y, Xt_all, W, WY_all):
    """
    Restricted model: A=0 => f_t = omega => rho_t constant.
    params = [omega, B(ignored), log_sigma, beta_reg(k)]
    """
    omega, log_sigma = params[0], params[2]
    beta_reg = params[3:]

    rho_c = link(omega)
    sigma2 = np.exp(2.0 * log_sigma)
    T, n = Y.shape

    I_n = np.eye(n)
    S = I_n - rho_c * W
    sign, logdetS = np.linalg.slogdet(S)
    if sign <= 0:
        return np.inf

    total_ll = 0.0
    for t in range(T):
        e_t = S @ Y[t] - Xt_all[t] @ beta_reg
        ll_t = (
            logdetS
            - (n / 2.0) * np.log(2.0 * np.pi * sigma2)
            - (e_t @ e_t) / (2.0 * sigma2)
        )
        if not np.isfinite(ll_t):
            return np.inf
        total_ll += ll_t

    return -total_ll


def compute_kappa_hat(omega_r, log_sig_r, beta_reg_r, Y, Xt_all, W, WY_all):
    """
    Scaling factor kappa_hat for scaled QLR (slides p.51-53).
    """
    T, n = Y.shape
    rho_r = link(omega_r)
    sigma2 = np.exp(2.0 * log_sig_r)

    I_n = np.eye(n)
    S = I_n - rho_r * W
    S_inv = np.linalg.inv(S)
    SinvW = S_inv @ W

    tr_1 = np.trace(SinvW)
    tr_2 = np.trace(SinvW @ SinvW)

    h_dot = link_dot(omega_r)
    h_ddot = link_ddot(omega_r)

    sum_sq = 0.0
    sum_hess = 0.0

    for t in range(T):
        e_t = S @ Y[t] - Xt_all[t] @ beta_reg_r
        d_rho = (WY_all[t] @ e_t) / sigma2 - tr_1
        score_t = d_rho * h_dot

        hess_t = (
            d_rho * h_ddot
            - ((Y[t] @ (W.T @ WY_all[t])) / sigma2 + tr_2) * (h_dot ** 2)
        )

        sum_sq += score_t ** 2
        sum_hess += hess_t

    kappa_hat = (-sum_hess / T) ** (-1) * (sum_sq / T)
    return kappa_hat


def qlr_test(ll_unrestricted, params_opt, Y, Xt_all, W, WY_all, bounds_restricted):
    """
    QLR test for H0: A=0. Returns scaled statistic and CVs (assignment Table 1).
    """
    omega_m, B_m, log_sig_m = params_opt[0], params_opt[1], params_opt[3]
    beta_reg_m = params_opt[4:]
    k = len(beta_reg_m)

    x0_candidates = [
        np.array([omega_m, B_m, log_sig_m] + list(beta_reg_m)),
        np.array([0.5, 0.90, np.log(3.0)] + [0.0] * k),
        np.array([1.0, 0.95, np.log(2.0)] + [0.0] * k),
    ]

    lo_r = [b[0] for b in bounds_restricted]
    hi_r = [b[1] for b in bounds_restricted]

    best_ll_r, best_params_r = -np.inf, None
    for x0 in x0_candidates:
        x0 = np.clip(x0, lo_r, hi_r)
        res = minimize(
            _neg_loglik_restricted, x0,
            args=(Y, Xt_all, W, WY_all),
            method="L-BFGS-B",
            bounds=bounds_restricted,
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7},
        )
        if -res.fun > best_ll_r:
            best_ll_r = -res.fun
            best_params_r = res.x.copy()

    ll_restricted = best_ll_r
    omega_r, log_sig_r = best_params_r[0], best_params_r[2]
    beta_reg_r = best_params_r[3:]

    QLRT = 2.0 * (ll_unrestricted - ll_restricted)
    kappa_hat = compute_kappa_hat(omega_r, log_sig_r, beta_reg_r, Y, Xt_all, W, WY_all)
    QLR_tilde = QLRT / kappa_hat

    # Table 1 critical values for B=[0,0.990], alpha_L=0
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
        "ll_restricted": ll_restricted,
        "QLRT": QLRT,
        "kappa_hat": kappa_hat,
        "QLR_tilde": QLR_tilde,
        "omega_r": omega_r,
        "rho_const": link(omega_r),
        "sigma_r": np.exp(log_sig_r),
        "beta_reg_r": beta_reg_r,
        "cv_10": cv_10,
        "cv_05": cv_05,
        "cv_01": cv_01,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------
# Student-t score-driven filter (Q2c, Blasques et al. 2016)
# ---------------------------------------------------------------------
def run_filter_student_t(omega, B, A, log_sigma, nu, beta_reg, Y, Xt_all, W, WY_all):
    """
    Student-t score-driven homogeneous SDM filter with Sigma = sigma^2 I_n.

    Uses robust weight:
        w_tilde = (1 + n/nu) / (1 + Q_t/nu)
    where Q_t = e_t'e_t / sigma^2
    (Blasques et al. 2016, Eq. (12)).

    params:
      omega, B, A, log_sigma, nu, beta_reg(k)
    """
    if nu <= 2.0:
        return -1e12, None, None

    T, n = Y.shape
    sigma2 = np.exp(2.0 * log_sigma)
    I_n = np.eye(n)

    f = np.empty(T)
    rho = np.empty(T)
    f[0] = omega
    rho[0] = link(f[0])

    total_ll = 0.0

    for t in range(T):
        S = I_n - rho[t] * W
        sign, logdetS = np.linalg.slogdet(S)
        if sign <= 0:
            return -1e12, f, rho

        e_t = S @ Y[t] - Xt_all[t] @ beta_reg
        Q_t = (e_t @ e_t) / sigma2

        # log-likelihood contribution: log|S| + log t-density with Sigma=sigma^2 I
        ll_const = (
            gammaln((nu + n) / 2.0)
            - gammaln(nu / 2.0)
            - (n / 2.0) * np.log(nu * np.pi * sigma2)
        )
        ll_t = logdetS + ll_const - ((nu + n) / 2.0) * np.log(1.0 + Q_t / nu)
        if not np.isfinite(ll_t):
            return -1e12, f, rho
        total_ll += ll_t

        if t < T - 1:
            # robust weight (paper Eq. (12))
            w_tilde = (1.0 + n / nu) / (1.0 + Q_t / nu)

            S_inv = np.linalg.solve(S, I_n)
            d_rho = (w_tilde * (WY_all[t] @ e_t) / sigma2) - np.trace(S_inv @ W)
            score_t = d_rho * link_dot(f[t])

            f[t + 1] = omega * (1.0 - B) + B * f[t] + A * score_t
            rho[t + 1] = link(f[t + 1])

    return total_ll, f, rho


def _neg_loglik_student_t(params, Y, Xt_all, W, WY_all):
    ll, _, _ = run_filter_student_t(params[0], params[1], params[2], params[3],
                                    params[4], params[5:], Y, Xt_all, W, WY_all)
    return -ll


def estimate_student_t_model(Y, Xt_all, W, WY_all, bounds, start_list):
    """
    MLE for Student-t model using multiple starting points.
    params = [omega, B, A, log_sigma, nu, beta_reg(k)]
    """
    best_params = None
    best_ll = -np.inf

    for x0 in start_list:
        res = minimize(
            _neg_loglik_student_t, x0,
            args=(Y, Xt_all, W, WY_all),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-7},
        )
        ll = -res.fun
        if ll > best_ll:
            best_ll = ll
            best_params = res.x.copy()

    return best_params, best_ll


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_rho_path(rho_main, rho_const, T, save_path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rho_main, linewidth=0.8, label=r'$\hat{\rho}_t$ (time-varying)')
    ax.axhline(y=rho_const, color="red", linestyle="--", linewidth=1.5,
               label=r"time-constant $\hat{\rho}$")
    ax.set_xlim(0, T)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\hat{\rho}_t$")
    ax.set_title("Gaussian score-driven spatial spillover dynamics")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_rho_comparison(rho_paths, labels, rho_const, save_path):
    n_specs = len(rho_paths)
    fig, axes = plt.subplots(1, n_specs, figsize=(5 * n_specs, 4), sharey=True)
    if n_specs == 1:
        axes = [axes]
    for ax, rho_path, label in zip(axes, rho_paths, labels):
        ax.plot(rho_path, linewidth=0.7)
        ax.axhline(y=rho_const, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Time")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylabel(r"$\hat{\rho}_t$")
    fig.suptitle("Comparison of alternative parameter space specifications", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_rho_gauss_vs_t(rho_gauss, rho_t, save_path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rho_gauss, linestyle="--", linewidth=1.0, label="Gaussian")
    ax.plot(rho_t, linestyle="-", linewidth=1.0, label="Student-t")
    ax.set_xlim(0, len(rho_gauss))
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\hat{\rho}_t$")
    ax.set_title("Filtered spatial dependence: Gaussian vs Student-t")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_rho_difference(rho_gauss, rho_t, save_path):
    """
    Plot difference in filtered spatial dependence:
        Δrho_t = rho_t(Student-t) - rho_t(Gaussian)

    This makes the robustness effect visually clearer:
    negative values often occur around extreme Gaussian spikes.
    """
    rho_gauss = np.asarray(rho_gauss)
    rho_t = np.asarray(rho_t)

    if rho_gauss.shape != rho_t.shape:
        raise ValueError("rho_gauss and rho_t must have the same length")

    diff = rho_t - rho_gauss

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(diff, linewidth=1.0, label=r"$\hat{\rho}_t^{(t)} - \hat{\rho}_t^{(G)}$")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlim(0, len(diff))
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Difference in $\hat{\rho}_t$")
    ax.set_title("Difference in filtered spatial dependence (Student-t minus Gaussian)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")

# ---------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------
def print_estimates_gauss(label, params, ll, rho_path):
    omega, B, A, log_sig = params[:4]
    beta_reg = params[4:]
    print(f"\n{label}")
    print(f"  omega   = {omega:.6f}")
    print(f"  B       = {B:.6f}")
    print(f"  A       = {A:.6f}")
    print(f"  sigma   = {np.exp(log_sig):.6f}")
    print(f"  beta_reg = [{', '.join(f'{b:.4f}' for b in beta_reg)}]")
    print(f"  Log-likelihood = {ll:.4f}")
    print(f"  rho_t range    = [{rho_path.min():.4f}, {rho_path.max():.4f}]")


def print_estimates_t(label, params, ll, rho_path):
    omega, B, A, log_sig, nu = params[:5]
    beta_reg = params[5:]
    print(f"\n{label}")
    print(f"  omega   = {omega:.6f}")
    print(f"  B       = {B:.6f}")
    print(f"  A       = {A:.6f}")
    print(f"  sigma   = {np.exp(log_sig):.6f}")
    print(f"  nu      = {nu:.6f}")
    print(f"  beta_reg = [{', '.join(f'{b:.4f}' for b in beta_reg)}]")
    print(f"  Log-likelihood = {ll:.4f}")
    print(f"  rho_t range    = [{rho_path.min():.4f}, {rho_path.max():.4f}]")


def print_qlr_results(r):
    print("\n" + "=" * 60)
    print("QLR TEST FOR TIME VARIATION  (H0: A = 0)")
    print("=" * 60)
    print(f"  Restricted model: rho_const = {r['rho_const']:.6f}  "
          f"(omega_r = {r['omega_r']:.6f}, sigma_r = {r['sigma_r']:.6f})")
    print(f"  LL (restricted)   = {r['ll_restricted']:.4f}")
    print(f"  QLRT   (raw)      = {r['QLRT']:.4f}")
    print(f"  kappa_hat         = {r['kappa_hat']:.6f}")
    print(f"  QLR_tilde         = {r['QLR_tilde']:.4f}")
    print("  Critical values (Table 1, B=[0,0.99], alpha_L=0):")
    print(f"    10%: {r['cv_10']},  5%: {r['cv_05']},  1%: {r['cv_01']}")
    print(f"  Conclusion: {r['conclusion']}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    CDS_PATH = str(ROOT / "data" / "cds_data.xlsx")
    WEIGHTS_PATH = str(ROOT / "data" / "cds_spatialweights.xlsx")

    Y, Xt_all, W, WY_all, _ = load_data(CDS_PATH, WEIGHTS_PATH)
    T, n = Y.shape
    k = K_REGRESSORS
    print(f"  T = {T},  n = {n},  k = {k}")

    # ------------------------------
    # Q2(a): Gaussian main spec
    # ------------------------------
    print("\nMAIN SPECIFICATION (Gaussian, slides p.55)")
    print("  B = [0, 0.99],  A = [0, 1], omega in [0,2], sigma in [1,10], beta_reg in [-10,10]")

    bounds_main = (
        [(0.0, 2.0), (0.0, 0.99), (0.0, 1.0), (np.log(1.0), np.log(10.0))]
        + [(-10.0, 10.0)] * k
    )

    params_main, ll_main, _ = estimate_gaussian_model(Y, Xt_all, W, WY_all, bounds_main)
    ll_check, f_main, rho_main = run_filter_gaussian(
        params_main[0], params_main[1], params_main[2], params_main[3],
        params_main[4:], Y, Xt_all, W, WY_all
    )
    print_estimates_gauss("Gaussian main estimates:", params_main, ll_main, rho_main)

    # Alternative specs (optional; keep if you need them)
    print("\nALTERNATIVE SPEC 1 (Gaussian): A in [0,2]")
    bounds_alt1 = (
        [(0.0, 2.0), (0.0, 0.99), (0.0, 2.0), (np.log(1.0), np.log(10.0))]
        + [(-10.0, 10.0)] * k
    )
    params_alt1, ll_alt1, _ = estimate_gaussian_model(Y, Xt_all, W, WY_all, bounds_alt1)
    _, _, rho_alt1 = run_filter_gaussian(
        params_alt1[0], params_alt1[1], params_alt1[2], params_alt1[3],
        params_alt1[4:], Y, Xt_all, W, WY_all
    )
    print_estimates_gauss("Gaussian alt1 estimates:", params_alt1, ll_alt1, rho_alt1)

    print("\nALTERNATIVE SPEC 2 (Gaussian): B in [0,0.999]")
    bounds_alt2 = (
        [(0.0, 2.0), (0.0, 0.999), (0.0, 1.0), (np.log(1.0), np.log(10.0))]
        + [(-10.0, 10.0)] * k
    )
    params_alt2, ll_alt2, _ = estimate_gaussian_model(Y, Xt_all, W, WY_all, bounds_alt2)
    _, _, rho_alt2 = run_filter_gaussian(
        params_alt2[0], params_alt2[1], params_alt2[2], params_alt2[3],
        params_alt2[4:], Y, Xt_all, W, WY_all
    )
    print_estimates_gauss("Gaussian alt2 estimates:", params_alt2, ll_alt2, rho_alt2)

    # QLR test for Q2(a)
    print("\nQLR TEST FOR TIME VARIATION (Gaussian, H0: A=0)")
    bounds_restricted = (
        [(0.0, 2.0), (0.0, 0.99), (np.log(1.0), np.log(10.0))]
        + [(-10.0, 10.0)] * k
    )
    qlr_results = qlr_test(ll_main, params_main, Y, Xt_all, W, WY_all, bounds_restricted)
    print_qlr_results(qlr_results)
    rho_const = qlr_results["rho_const"]

    # Figures for Q2(a)
    print("\nGenerating Q2(a) figures ...")
    plot_rho_path(rho_main, rho_const, T, save_path=str(FIG_DIR / "q2a_rho_main.png"))
    plot_rho_comparison(
        [rho_main, rho_alt1, rho_alt2],
        [
            f"Main: B=[0,0.99], A in [0,1]\nA={params_main[2]:.4f}, B={params_main[1]:.4f}",
            f"Alt 1: A in [0,2]\nA={params_alt1[2]:.4f}, B={params_alt1[1]:.4f}",
            f"Alt 2: B=[0,0.999]\nA={params_alt2[2]:.4f}, B={params_alt2[1]:.4f}",
        ],
        rho_const,
        save_path=str(FIG_DIR / "q2a_rho_comparison.png"),
    )

    # ------------------------------
    # Q2(c): Student-t model + comparison plot
    # ------------------------------
    print("\nSTUDENT-t SPECIFICATION (Q2c, Blasques et al. 2016)")
    print("  Same h(f)=0.95*tanh(f) and same regressors; Student-t errors with df nu>2")

    # Student-t bounds (choose nu upper bound large enough; keep nu>2)
    bounds_t = (
        [(0.0, 2.0), (0.0, 0.99), (0.0, 1.0), (np.log(1.0), np.log(10.0)), (2.05, 100.0)]
        + [(-10.0, 10.0)] * k
    )

    # Build starting points from Gaussian main estimates + multiple nu guesses
    omega0, B0, A0, log_sig0 = params_main[0], params_main[1], params_main[2], params_main[3]
    beta0 = params_main[4:]
    nu_guesses = [3.0, 5.0, 10.0, 30.0]
    starts_t = []
    for nu0 in nu_guesses:
        starts_t.append(np.r_[omega0, B0, A0, log_sig0, nu0, beta0])

    # Add a couple of generic starts too
    starts_t.append(np.r_[0.8, 0.95, 0.05, np.log(3.0), 5.0, np.zeros(k)])
    starts_t.append(np.r_[1.2, 0.90, 0.10, np.log(2.0), 10.0, np.zeros(k)])

    params_t, ll_t = estimate_student_t_model(Y, Xt_all, W, WY_all, bounds_t, starts_t)
    ll_check_t, f_t, rho_t = run_filter_student_t(
        params_t[0], params_t[1], params_t[2], params_t[3], params_t[4], params_t[5:],
        Y, Xt_all, W, WY_all
    )
    print_estimates_t("Student-t estimates:", params_t, ll_t, rho_t)

    # Model comparison metrics
    p_gauss = 4 + k  # omega,B,A,log_sigma + beta_reg(k)
    p_t = 5 + k      # add nu
    aic_gauss = -2.0 * ll_main + 2.0 * p_gauss
    aic_t = -2.0 * ll_t + 2.0 * p_t
    bic_gauss = -2.0 * ll_main + np.log(T) * p_gauss
    bic_t = -2.0 * ll_t + np.log(T) * p_t

    print("\nMODEL COMPARISON (Q2c)")
    print(f"  Gaussian:   LL={ll_main:.4f},  AIC={aic_gauss:.2f},  BIC={bic_gauss:.2f},  p={p_gauss}")
    print(f"  Student-t:  LL={ll_t:.4f},     AIC={aic_t:.2f},      BIC={bic_t:.2f},      p={p_t}")
    print(f"  LL improvement (t - Gauss) = {ll_t - ll_main:.4f}")

    # Figure-4 style overlay plot
    print("\nGenerating Q2(c) figure ...")
    plot_rho_gauss_vs_t(rho_main, rho_t, save_path=str(FIG_DIR / "q2c_rho_gauss_vs_t.png"))
    # Difference plot (helps visualize robustness effect)
    plot_rho_difference(
        rho_main,
        rho_t,
        save_path=str(FIG_DIR / "q2c_rho_difference_t_minus_gauss.png"),
    )

    print("\nDONE. Files saved to:", FIG_DIR)


if __name__ == "__main__":
    main()