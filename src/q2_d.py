# -*- coding: utf-8 -*-
"""
Question 2(d): QLR test with multivariate t score-driven dynamics
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt

from q2_a import (load_data, link, link_dot, link_ddot,
                   K_REGRESSORS, RHO_MAX)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)


# Multivariate t score-driven filter

def run_t_filter(omega, B, alpha, sigma2, nu, beta_reg,
                 Y, Xt_all, W, WY_all):
    """
    Multivariate t score-driven filter for the spatial spillover parameter.
    """
    T, n = Y.shape
    I_n = np.eye(n)

    f = np.empty(T)
    rho = np.empty(T)
    f[0] = omega
    rho[0] = link(omega)
    total_ll = 0.0

    const = (gammaln((nu + n) / 2.0) - gammaln(nu / 2.0)
             - (n / 2.0) * np.log(nu * np.pi)
             - (n / 2.0) * np.log(sigma2))

    for t in range(T):
        S_t = I_n - rho[t] * W
        sign, logdet_S = np.linalg.slogdet(S_t)
        if sign <= 0:
            return -1e10, f, rho

        e_t = S_t @ Y[t] - Xt_all[t] @ beta_reg
        Q_t = np.dot(e_t, e_t) / sigma2

        ll_t = const + logdet_S - ((nu + n) / 2.0) * np.log(1.0 + Q_t / nu)

        if not np.isfinite(ll_t):
            return -1e10, f, rho
        total_ll += ll_t

        if t < T - 1:
            S_inv = np.linalg.solve(S_t, I_n)
            w_t = (nu + n) / (nu + Q_t)
            d_rho = (w_t * np.dot(WY_all[t], e_t) / sigma2
                     - np.trace(S_inv @ W))
            score_t = d_rho * link_dot(f[t])
            f[t + 1] = omega * (1.0 - B) + B * f[t] + alpha * score_t
            rho[t + 1] = link(f[t + 1])

    return total_ll, f, rho


def _neg_loglik_t(params, Y, Xt_all, W, WY_all):
    sigma2 = np.exp(2.0 * params[3])
    nu = np.exp(params[4])
    ll, _, _ = run_t_filter(params[0], params[1], params[2],
                            sigma2, nu, params[5:],
                            Y, Xt_all, W, WY_all)
    return -ll


def _neg_loglik_t_restricted(params, Y, Xt_all, W, WY_all):
    """Restricted """
    omega = params[0]
    sigma2 = np.exp(2.0 * params[2])
    nu = np.exp(params[3])
    beta_reg = params[4:]

    rho_c = link(omega)
    T, n = Y.shape
    I_n = np.eye(n)
    S = I_n - rho_c * W

    sign, logdet_S = np.linalg.slogdet(S)
    if sign <= 0:
        return np.inf

    const = (gammaln((nu + n) / 2.0) - gammaln(nu / 2.0)
             - (n / 2.0) * np.log(nu * np.pi)
             - (n / 2.0) * np.log(sigma2))

    total_ll = 0.0
    for t in range(T):
        e_t = S @ Y[t] - Xt_all[t] @ beta_reg
        Q_t = np.dot(e_t, e_t) / sigma2
        ll_t = const + logdet_S - ((nu + n) / 2.0) * np.log(1.0 + Q_t / nu)
        if not np.isfinite(ll_t):
            return np.inf
        total_ll += ll_t

    return -total_ll


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def estimate_t_model(Y, Xt_all, W, WY_all, bounds, n_starts=6):
    """Estimate the unrestricted multivariate-t score-driven model (MLE)."""
    k = K_REGRESSORS
    starts = [
        [0.5,  0.95, 0.05, np.log(3.0), np.log(10.0)] + [0.0] * k,
        [1.0,  0.90, 0.10, np.log(2.0), np.log(5.0)]  + [0.0] * k,
        [0.5,  0.98, 0.02, np.log(5.0), np.log(8.0)]  + [0.0] * k,
        [1.5,  0.95, 0.05, np.log(3.0), np.log(15.0)] + [0.0] * k,
        [0.5,  0.85, 0.20, np.log(3.0), np.log(20.0)] + [0.0] * k,
        [1.0,  0.97, 0.01, np.log(4.0), np.log(7.0)]  + [0.0] * k,
    ]

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    best_params, best_ll = None, -np.inf
    results_list = []

    for x0 in starts[:n_starts]:
        x0 = np.clip(x0, lo, hi)
        res = minimize(
            _neg_loglik_t, x0,
            args=(Y, Xt_all, W, WY_all),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7},
        )
        ll = -res.fun
        results_list.append((ll, res.x))
        if ll > best_ll:
            best_ll = ll
            best_params = res.x.copy()

    return best_params, best_ll, results_list


def compute_kappa_hat_t(omega_r, log_sig_r, log_nu_r, beta_reg_r,
                        Y, Xt_all, W, WY_all):
    """Compute the scaling factor kappa_hat at the restricted estimates."""
    T, n = Y.shape
    rho_r = link(omega_r)
    sigma2 = np.exp(2.0 * log_sig_r)
    nu = np.exp(log_nu_r)
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
        Q_t = np.dot(e_t, e_t) / sigma2
        w_t = (nu + n) / (nu + Q_t)

        Wy_e = np.dot(WY_all[t], e_t)
        d_rho = w_t * Wy_e / sigma2 - tr_1

        score_t = d_rho * h_dot

        WyWy = np.dot(WY_all[t], WY_all[t])
        hess_rho = (2.0 * w_t ** 2 / ((nu + n) * sigma2 ** 2) * Wy_e ** 2
                    - w_t * WyWy / sigma2
                    - tr_2)
        hess_t = d_rho * h_ddot + hess_rho * h_dot ** 2

        sum_sq += score_t ** 2
        sum_hess += hess_t

    kappa_hat = (-sum_hess / T) ** (-1) * (sum_sq / T)
    return kappa_hat


# ---------------------------------------------------------------------------
# QLR test
# ---------------------------------------------------------------------------

def qlr_test_t(ll_unrestricted, params_opt, Y, Xt_all, W, WY_all,
               bounds_restricted, beta_U=0.99, alpha_L=0):
    """
    QLR test for absence of score-driven dynamics
    """
    k = K_REGRESSORS
    omega_m = params_opt[0]
    log_sig_m = params_opt[3]
    log_nu_m = params_opt[4]
    beta_reg_m = params_opt[5:]

    x0_candidates = [
        np.array([omega_m, 0.9, log_sig_m, log_nu_m] + list(beta_reg_m)),
        np.array([0.5,  0.9,  np.log(3.0), np.log(10.0)] + [0.0] * k),
        np.array([1.0,  0.95, np.log(2.0), np.log(5.0)]  + [0.0] * k),
        np.array([1.5,  0.85, np.log(4.0), np.log(15.0)] + [0.0] * k),
    ]

    lo_r = [b[0] for b in bounds_restricted]
    hi_r = [b[1] for b in bounds_restricted]

    best_ll_r, best_params_r = -np.inf, None
    for x0 in x0_candidates:
        x0 = np.clip(x0, lo_r, hi_r)
        res = minimize(
            _neg_loglik_t_restricted, x0,
            args=(Y, Xt_all, W, WY_all),
            method='L-BFGS-B',
            bounds=bounds_restricted,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7},
        )
        if -res.fun > best_ll_r:
            best_ll_r = -res.fun
            best_params_r = res.x.copy()

    ll_restricted = best_ll_r
    omega_r = best_params_r[0]
    log_sig_r = best_params_r[2]
    log_nu_r = best_params_r[3]
    beta_reg_r = best_params_r[4:]

    QLRT = 2.0 * (ll_unrestricted - ll_restricted)
    kappa_hat = compute_kappa_hat_t(omega_r, log_sig_r, log_nu_r, beta_reg_r,
                                    Y, Xt_all, W, WY_all)
    QLR_tilde = QLRT / kappa_hat

    # critical values from Table 1 of the assignment
    cv_10, cv_05, cv_01 = 3.266, 4.613, 7.696

    if QLR_tilde >= cv_01:
        conclusion = "Reject H0 at 1%: strong evidence of time-varying rho_t."
    elif QLR_tilde >= cv_05:
        conclusion = "Reject H0 at 5%."
    elif QLR_tilde >= cv_10:
        conclusion = "Reject H0 at 10%."
    else:
        conclusion = "Fail to reject H0: no evidence of time variation."

    return {
        'll_restricted': ll_restricted,
        'QLRT': QLRT,
        'kappa_hat': kappa_hat,
        'QLR_tilde': QLR_tilde,
        'omega_r': omega_r,
        'rho_const': link(omega_r),
        'sigma_r': np.exp(log_sig_r),
        'nu_r': np.exp(log_nu_r),
        'beta_reg_r': beta_reg_r,
        'cv_10': cv_10,
        'cv_05': cv_05,
        'cv_01': cv_01,
        'conclusion': conclusion,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_t_estimates(label, params, ll, rho_path):
    omega, B_p, alpha, log_sig, log_nu = params[:5]
    beta_reg = params[5:]
    print(f"\n{label}")
    print(f"  omega    = {omega:.6f}")
    print(f"  B        = {B_p:.6f}")
    print(f"  alpha    = {alpha:.6f}")
    print(f"  sigma    = {np.exp(log_sig):.6f}")
    print(f"  nu       = {np.exp(log_nu):.4f}")
    print(f"  beta_reg = [{', '.join(f'{b:.4f}' for b in beta_reg)}]")
    print(f"  Log-lik  = {ll:.4f}")
    print(f"  rho range = [{rho_path.min():.4f}, {rho_path.max():.4f}]")


def print_qlr_results_t(r):
    print("\n" + "=" * 60)
    print("QLR TEST (multivariate t dynamics)  H0: alpha = 0")
    print("=" * 60)
    print(f"  Restricted: rho_const = {r['rho_const']:.6f}  "
          f"(omega = {r['omega_r']:.6f}, sigma = {r['sigma_r']:.6f}, "
          f"nu = {r['nu_r']:.4f})")
    print(f"  LL (restricted) = {r['ll_restricted']:.4f}")
    print(f"  QLRT (raw)      = {r['QLRT']:.4f}")
    print(f"  kappa_hat       = {r['kappa_hat']:.6f}")
    print(f"  QLR_tilde       = {r['QLR_tilde']:.4f}")
    print(f"  Critical values (Table 1, B=[0,0.99], Theta_alpha=[0,1], alpha_L=0):")
    print(f"    10%: {r['cv_10']},  5%: {r['cv_05']},  1%: {r['cv_01']}")
    print(f"  Conclusion: {r['conclusion']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """analysis for Question 2(d)."""
    print("=" * 60)
    print("QUESTION 2(d): QLR test with multivariate t dynamics")
    print("=" * 60)

    CDS_PATH = str(ROOT / "data" / "cds_data.xlsx")
    WEIGHTS_PATH = str(ROOT / "data" / "cds_spatialweights.xlsx")
    Y, Xt_all, W, WY_all, _ = load_data(CDS_PATH, WEIGHTS_PATH)
    T, n = Y.shape
    k = K_REGRESSORS
    print(f"  T = {T},  n = {n},  k = {k}")

    print("\nSpecification:")
    print("  B = [0, 0.99],  Theta_alpha = [0, 1]")
    print("  h(f) = 0.95 * tanh(f)")
    print("  sigma in [1, 10],  nu in [3, 200]")

    bounds_unr = (
        [(0.0, 2.0),                        # omega
         (0.0, 0.99),                        # B
         (0.0, 1.0),                         # alpha
         (np.log(1.0), np.log(10.0)),        # log_sigma
         (np.log(3.0), np.log(200.0))]       # log_nu
        + [(-10.0, 10.0)] * k               # beta_reg
    )

    # ── Unrestricted estimation ──
    print("\nEstimating unrestricted t score-driven model ...")
    params_opt, ll_opt, _ = estimate_t_model(Y, Xt_all, W, WY_all, bounds_unr)

    sigma2_opt = np.exp(2.0 * params_opt[3])
    nu_opt = np.exp(params_opt[4])
    _, f_opt, rho_opt = run_t_filter(
        params_opt[0], params_opt[1], params_opt[2],
        sigma2_opt, nu_opt, params_opt[5:],
        Y, Xt_all, W, WY_all,
    )

    print_t_estimates("Unrestricted t model estimates:", params_opt, ll_opt, rho_opt)

    # ── QLR test (H0: alpha = 0) ──
    print("\nRunning QLR test (H0: alpha = 0) ...")
    bounds_restr = (
        [(0.0, 2.0),                        # omega
         (0.0, 0.99),                        # B (irrelevant under H0)
         (np.log(1.0), np.log(10.0)),        # log_sigma
         (np.log(3.0), np.log(200.0))]       # log_nu
        + [(-10.0, 10.0)] * k               # beta_reg
    )

    qlr = qlr_test_t(
        ll_opt, params_opt, Y, Xt_all, W, WY_all,
        bounds_restr, beta_U=0.99, alpha_L=0,
    )

    print_qlr_results_t(qlr)

    # ── Figure: estimated ρ_t path ──
    rho_const = qlr['rho_const']
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rho_opt, color='darkgreen', linewidth=0.8,
            label=r'$\hat{\rho}_t$ (t score-driven)')
    ax.axhline(y=rho_const, color='red', linestyle='--', linewidth=1.5,
               label=r'time-constant $\hat{\rho}$')
    ax.set_xlim(0, T)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\hat{\rho}_t$', fontsize=12)
    ax.set_title('Multivariate t score-driven spatial spillover dynamics',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    save_path = str(FIG_DIR / 'q2d_rho_t_dynamics.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n  Saved: {save_path}")
    plt.show()
    plt.close()

    print("\nQUESTION 2(d) COMPLETE")
    return params_opt, ll_opt, rho_opt, qlr


if __name__ == "__main__":
    main()
