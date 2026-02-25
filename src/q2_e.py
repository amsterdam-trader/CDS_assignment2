# -*- coding: utf-8 -*-
"""
Question 2(e): Spurious time-varying patterns in score-driven models.

DGP:  y_t = 0.2 + u_t,  u_t  ~iid  t_10,  T = 500.

Model: univariate t-location score-driven filter
    y_t | f_t  ~  t_v(f_t, sigma²)
    f_{t+1} = ω(1-B) + B f_t + alpha s_t,   f_1 = ω
    s_t = (v+1)(y_t − f_t) / (v*sigma² + (y_t - f_t)²)

Parameter spaces: B ⊂ [0, 1], alpha = [0, 2], omega ⊂ [-2, 2], sigma ⊂ [0.1, 10], nu ⊂ [3, 200] .
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt
import warnings

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Univariate t-location score-driven filter
# ---------------------------------------------------------------------------

def run_t_location_filter(omega, B, alpha, sigma2, nu, Y):
    """
    Univariate t-location score-driven filter.
    """
    T = len(Y)
    f = np.empty(T)
    f[0] = omega
    total_ll = 0.0

    const = (gammaln((nu + 1) / 2.0) - gammaln(nu / 2.0)
             - 0.5 * np.log(nu * np.pi * sigma2))

    for t in range(T):
        resid = Y[t] - f[t]
        Q_t = resid ** 2 / sigma2

        ll_t = const - ((nu + 1) / 2.0) * np.log(1.0 + Q_t / nu)
        if not np.isfinite(ll_t):
            return -1e10, f
        total_ll += ll_t

        if t < T - 1:
            score_t = (nu + 1) * resid / (nu * sigma2 + resid ** 2)
            f[t + 1] = omega * (1.0 - B) + B * f[t] + alpha * score_t

    return total_ll, f


# ---------------------------------------------------------------------------
# Optimisation wrapper
# ---------------------------------------------------------------------------

def _neg_loglik(params, Y):
    """params = [omega, B, alpha, log_sigma, log_nu]"""
    sigma2 = np.exp(2.0 * params[3])
    nu = np.exp(params[4])
    ll, _ = run_t_location_filter(params[0], params[1], params[2],
                                  sigma2, nu, Y)
    return -ll


def estimate_t_location(Y, bounds, n_starts=6):
    """Estimate the univariate t-location score-driven model by MLE."""
    starts = [
        [0.0,  0.90, 0.05, np.log(1.0), np.log(10.0)],
        [0.2,  0.95, 0.10, np.log(1.5), np.log(8.0)],
        [0.0,  0.98, 0.01, np.log(0.5), np.log(15.0)],
        [0.5,  0.80, 0.20, np.log(1.0), np.log(5.0)],
        [-0.1, 0.85, 0.50, np.log(2.0), np.log(12.0)],
        [0.3,  0.97, 0.02, np.log(1.0), np.log(20.0)],
    ]

    lo = [b[0] for b in bounds]
    hi = [b[1] for b in bounds]

    best_params, best_ll = None, -np.inf

    for x0 in starts[:n_starts]:
        x0 = np.clip(x0, lo, hi)
        res = minimize(
            _neg_loglik, x0,
            args=(Y,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7},
        )
        ll = -res.fun
        if ll > best_ll:
            best_ll = ll
            best_params = res.x.copy()

    return best_params, best_ll


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Demonstrate that score-driven models may produce spurious
    time-varying patterns even when the true parameter is constant.
    """
    print("=" * 60)
    print("QUESTION 2(e): Spurious time variation in score-driven models")
    print("=" * 60)

    T = 500
    mu_true = 0.2
    nu_true = 10
    n_runs = 4

    print(f"\nDGP:  y_t = {mu_true} + u_t,  u_t ~ t_{nu_true},  T = {T}")
    print(f"\nModel: univariate t-location score-driven filter (S_t = 1)")
    print(f"  f_{{t+1}} = omega(1-B) + B f_t + alpha s_t,   f_1 = omega")
    print(f"  s_t = (nu+1)(y_t - f_t) / (nu sigma^2 + (y_t - f_t)^2)")
    print(f"\nParameter spaces:")
    print(f"  B        in [0, 0.999]")
    print(f"  Theta_alpha = [0, 2]")
    print(f"  omega    in [-2, 2]")
    print(f"  sigma    in [0.1, 10]   (via log)")
    print(f"  nu       in [3, 200]    (via log)")
    print(f"\nNumber of simulation runs: {n_runs}")

    bounds = [
        (-2.0, 2.0),                      # omega
        (0.0,  0.999),                     # B
        (0.0,  2.0),                       # alpha
        (np.log(0.1), np.log(10.0)),       # log_sigma
        (np.log(3.0), np.log(200.0)),      # log_nu
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for i, ax in enumerate(axes.flat):
        np.random.seed(42 + i * 137)
        Y = mu_true + np.random.standard_t(df=nu_true, size=T)

        params, ll = estimate_t_location(Y, bounds)
        omega_est, B_est, alpha_est = params[0], params[1], params[2]
        sigma_est = np.exp(params[3])
        nu_est = np.exp(params[4])

        _, f_path = run_t_location_filter(
            omega_est, B_est, alpha_est, sigma_est ** 2, nu_est, Y
        )

        ax.plot(f_path, color='steelblue', linewidth=0.8,
                label=r'$\hat{f}_t$')
        ax.axhline(y=mu_true, color='red', linestyle='--', linewidth=1.5,
                   label=fr'True $\mu$ = {mu_true}')
        ax.set_title(
            fr'Run {i+1}: $\hat{{\alpha}}$={alpha_est:.3f}, '
            fr'$\hat{{B}}$={B_est:.3f}, '
            fr'$\hat{{\sigma}}$={sigma_est:.3f}, '
            fr'$\hat{{\nu}}$={nu_est:.1f}',
            fontsize=10,
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)

        print(f"\n  Run {i+1}:  omega={omega_est:.4f}  B={B_est:.4f}  "
              f"alpha={alpha_est:.4f}  sigma={sigma_est:.4f}  "
              f"nu={nu_est:.2f}  LL={ll:.2f}")

    for ax in axes[1]:
        ax.set_xlabel('Time', fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\hat{f}_t$', fontsize=11)

    fig.suptitle(
        fr'Spurious time variation: true location is constant at $\mu$ = {mu_true}',
        fontsize=13,
    )
    plt.tight_layout()
    save_path = str(FIG_DIR / 'q2e_spurious_time_variation.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n  Saved: {save_path}")
    plt.show()
    plt.close()

    print("\n" + "-" * 60)
    print("Observation:")
    print("  Despite the true location being constant (mu = 0.2),")
    print("  the estimated f_t paths show pronounced time-varying patterns.")
    print("  This occurs because the optimizer finds alpha > 0 (and B close")
    print("  to 1) to improve the in-sample likelihood by fitting noise.")
    print("  The score-driven update chases random fluctuations in y_t,")
    print("  producing a spurious time-varying path.")
    print("  This highlights the need for a formal QLR test before")
    print("  concluding that time variation is genuinely present.")
    print("-" * 60)
    print("\nQUESTION 2(e) COMPLETE")


if __name__ == "__main__":
    main()
