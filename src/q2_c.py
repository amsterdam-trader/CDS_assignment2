# -*- coding: utf-8 -*-
"""
Question 2(c): Student-t score-driven homogeneous SDM (Blasques et al. 2016)

Estimates the Student-t score-driven model on real CDS data and produces
comparison plots (Gaussian vs Student-t filtered rho).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from q2_a import (
    link, link_dot, main_q2a,
    K_REGRESSORS, FIG_DIR,
)


# ---------------------------------------------------------------------
# Student-t score-driven filter (Blasques et al. 2016)
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
        delta_rho_t = rho_t(Student-t) - rho_t(Gaussian)

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(gauss_results=None):
    """
    Run Q2(c): Student-t score-driven model + Gaussian vs Student-t comparison.
    If *gauss_results* is None, runs main_q2a() first (slow).
    """
    if gauss_results is None:
        gauss_results = main_q2a()

    params_main = gauss_results["params_main"]
    ll_main = gauss_results["ll_main"]
    rho_main = gauss_results["rho_main"]
    Y = gauss_results["Y"]
    Xt_all = gauss_results["Xt_all"]
    W = gauss_results["W"]
    WY_all = gauss_results["WY_all"]

    T, n = Y.shape
    k = K_REGRESSORS

    print("\nSTUDENT-t SPECIFICATION (Q2c, Blasques et al. 2016)")
    print("  Same h(f)=0.95*tanh(f) and same regressors; Student-t errors with df nu>2")

    bounds_t = (
        [(0.0, 2.0), (0.0, 0.99), (0.0, 1.0), (np.log(1.0), np.log(10.0)), (2.05, 100.0)]
        + [(-10.0, 10.0)] * k
    )

    omega0, B0, A0, log_sig0 = params_main[0], params_main[1], params_main[2], params_main[3]
    beta0 = params_main[4:]
    nu_guesses = [3.0, 5.0, 10.0, 30.0]
    starts_t = []
    for nu0 in nu_guesses:
        starts_t.append(np.r_[omega0, B0, A0, log_sig0, nu0, beta0])
    starts_t.append(np.r_[0.8, 0.95, 0.05, np.log(3.0), 5.0, np.zeros(k)])
    starts_t.append(np.r_[1.2, 0.90, 0.10, np.log(2.0), 10.0, np.zeros(k)])

    params_t, ll_t = estimate_student_t_model(Y, Xt_all, W, WY_all, bounds_t, starts_t)
    ll_check_t, f_t, rho_t = run_filter_student_t(
        params_t[0], params_t[1], params_t[2], params_t[3], params_t[4], params_t[5:],
        Y, Xt_all, W, WY_all
    )
    print_estimates_t("Student-t estimates:", params_t, ll_t, rho_t)

    p_gauss = 4 + k
    p_t = 5 + k
    aic_gauss = -2.0 * ll_main + 2.0 * p_gauss
    aic_t = -2.0 * ll_t + 2.0 * p_t
    bic_gauss = -2.0 * ll_main + np.log(T) * p_gauss
    bic_t = -2.0 * ll_t + np.log(T) * p_t

    print("\nMODEL COMPARISON (Q2c)")
    print(f"  Gaussian:   LL={ll_main:.4f},  AIC={aic_gauss:.2f},  BIC={bic_gauss:.2f},  p={p_gauss}")
    print(f"  Student-t:  LL={ll_t:.4f},     AIC={aic_t:.2f},      BIC={bic_t:.2f},      p={p_t}")
    print(f"  LL improvement (t - Gauss) = {ll_t - ll_main:.4f}")

    print("\nGenerating Q2(c) figure ...")
    plot_rho_gauss_vs_t(rho_main, rho_t, save_path=str(FIG_DIR / "q2c_rho_gauss_vs_t.png"))
    plot_rho_difference(
        rho_main,
        rho_t,
        save_path=str(FIG_DIR / "q2c_rho_difference_t_minus_gauss.png"),
    )

    print("\nQUESTION 2(c) COMPLETE")


if __name__ == "__main__":
    main()
