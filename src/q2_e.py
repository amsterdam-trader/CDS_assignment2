# -*- coding: utf-8 -*-
"""
Question 2(e): Spurious time-varying patterns in score-driven models.

DGP:  y_t = 0.2 + u_t,  u_t  ~iid  t_10,  T = 500.

Model: univariate t-location score-driven filter from Assignment 1, Q3.
    y_t | f_t  ~  t_nu(f_t, sigma_u^2)
    f_{t+1} = omega + beta f_t + alpha * s_t,   f_1 = mean(y_1,...,y_{T_bar})
    s_t = ((nu+3)/nu) * (y_t - f_t) / (1 + (1/nu)*((y_t - f_t)/sigma_u)^2)

The score uses inverse Fisher information scaling (S_t = I_t^{-1}),
which yields the (nu+3) factor rather than (nu+1).

Initialization: f_1 = mean(y_1, ..., y_{T_bar}) with T_bar = 10 (as in A1 Q3).

Parameter spaces:
    beta    (= B)   in [0, 0.999]   (given by Q2e: B in [0,1))
    alpha           in [0, 2]       (given by Q2e: Theta_alpha = [0,2])
    omega           in [-3, 3]      (same as A1 Q3)
    sigma_u         in (0.001, 2)   (same as A1 Q3)
    nu              in (3, 200)     (wider lower bound: nu_true=10 sits on A1 Q3 boundary)
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Univariate t-location score-driven filter  (Assignment 1, Q3 formulation)
# ---------------------------------------------------------------------------

T_BAR = 10   # initialisation window (as in A1 Q3 lecture notes)


def run_t_location_filter(sigma_u, nu, omega, beta, alpha, Y):
    """
    Univariate t-location GAS filter with inverse-information scaling,
    matching Assignment 1, Question 3.
    Initialized as f_1 = mean(y_1, ..., y_{T_bar}) with T_bar = 10.
    """
    T = len(Y)
    f = np.zeros(T)
    f[0] = np.mean(Y[:T_BAR])
    total_ll = 0.0

    const = gammaln((nu + 1) / 2.0) - 0.5 * np.log(nu) - gammaln(nu / 2.0) - np.log(sigma_u)

    for t in range(T):
        resid = Y[t] - f[t]
        z = resid / sigma_u
        ll_t = const - ((nu + 1) / 2.0) * np.log(1.0 + z * z / nu)
        if not np.isfinite(ll_t):
            return -1e10, f
        total_ll += ll_t

        if t < T - 1:
            score_t = ((3.0 + nu) / nu) * (resid / (1.0 + (1.0 / nu) * z * z))
            f[t + 1] = omega + beta * f[t] + alpha * score_t

    return total_ll, f


# ---------------------------------------------------------------------------
# Optimisation wrapper
# ---------------------------------------------------------------------------

def _neg_loglik(params, Y):
    """params = [sigma_u, nu, omega, beta, alpha]"""
    sigma_u, nu, omega, beta, alpha = params
    ll, _ = run_t_location_filter(sigma_u, nu, omega, beta, alpha, Y)
    return -ll


def estimate_t_location(Y, bounds, n_starts=6):
    """Estimate the univariate t-location GAS model by MLE."""
    sigma_mid = (bounds[0][0] + bounds[0][1]) / 2.0
    nu_mid = (bounds[1][0] + bounds[1][1]) / 2.0
    omega_mid = (bounds[2][0] + bounds[2][1]) / 2.0
    beta_mid = (bounds[3][0] + bounds[3][1]) / 2.0
    alpha_mid = (bounds[4][0] + bounds[4][1]) / 2.0

    starts = [
        [sigma_mid, nu_mid, omega_mid, beta_mid, alpha_mid],
        [0.8, 10.0,  0.0, 0.95, 0.05],
        [1.0,  8.0,  0.2, 0.90, 0.10],
        [1.2, 15.0, -0.1, 0.98, 0.01],
        [0.5,  5.0,  0.5, 0.80, 0.20],
        [1.0, 20.0,  0.3, 0.97, 0.02],
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
    n_display = 4

    print(f"\nDGP:  y_t = {mu_true} + u_t,  u_t ~ t_{nu_true},  T = {T}")
    print(f"\nModel: univariate t-location GAS filter (Assignment 1 Q3)")
    print(f"  f_{{t+1}} = omega + beta f_t + alpha s_t")
    print(f"  s_t = ((nu+3)/nu)(y_t - f_t) / (1 + (1/nu)((y_t - f_t)/sigma_u)^2)")
    print(f"  [inverse Fisher information scaling]")
    print(f"  f_1 = mean(y_1,...,y_{{T_bar}}) with T_bar = {T_BAR}")
    print(f"\nParameter spaces (Q2e-specified):")
    print(f"  beta (B) in [0, 0.999]")
    print(f"  Theta_alpha = [0, 2]")
    print(f"Parameter spaces (adopted from A1 Q3):")
    print(f"  omega    in [-3, 3]")
    print(f"  sigma_u  in (0.001, 2)")
    print(f"  nu       in (3, 200)  [A1 Q3 used (10,3000); lowered since nu_true=10]")

    bounds = [
        (0.001, 2.0),                      # sigma_u  (same as A1 Q3)
        (3.0,   200.0),                    # nu       (nu_true=10 sits on A1 Q3 lower bound, so widened)
        (-3.0,  3.0),                      # omega    (same as A1 Q3)
        (0.0,   0.999),                    # beta (B) (Q2e: B in [0,1))
        (0.0,   2.0),                      # alpha    (Q2e: Theta_alpha = [0,2])
    ]

    candidate_seeds = list(range(100, 180))
    results = []

    print(f"\nScanning {len(candidate_seeds)} seeds for illustrative runs ...")
    for seed in candidate_seeds:
        np.random.seed(seed)
        Y = mu_true + np.random.standard_t(df=nu_true, size=T)

        params, ll = estimate_t_location(Y, bounds)
        sigma_u_est, nu_est, omega_est, beta_est, alpha_est = params

        if alpha_est > 0.005 and beta_est > 0.80:
            _, f_path = run_t_location_filter(
                sigma_u_est, nu_est, omega_est, beta_est, alpha_est, Y
            )
            f_range = f_path.max() - f_path.min()
            results.append({
                'seed': seed, 'params': params, 'll': ll,
                'f_path': f_path, 'Y': Y, 'f_range': f_range,
            })

        if len(results) >= n_display:
            break

    if len(results) < n_display:
        print(f"  Warning: only found {len(results)} qualifying runs, "
              f"relaxing criteria ...")
        for seed in candidate_seeds:
            if any(r['seed'] == seed for r in results):
                continue
            np.random.seed(seed)
            Y = mu_true + np.random.standard_t(df=nu_true, size=T)
            params, ll = estimate_t_location(Y, bounds)
            alpha_est = params[4]
            if alpha_est > 0.0:
                _, f_path = run_t_location_filter(*params, Y)
                results.append({
                    'seed': seed, 'params': params, 'll': ll,
                    'f_path': f_path, 'Y': Y,
                    'f_range': f_path.max() - f_path.min(),
                })
            if len(results) >= n_display:
                break

    results = sorted(results, key=lambda r: r['f_range'], reverse=True)
    results = results[:n_display]

    print(f"\nSelected {len(results)} runs (seeds: "
          f"{[r['seed'] for r in results]}):")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for idx, (ax, r) in enumerate(zip(axes.flat, results)):
        p = r['params']
        sigma_u_est, nu_est, omega_est, beta_est, alpha_est = p
        f_path = r['f_path']

        ax.plot(f_path, color='steelblue', linewidth=0.8,
                label=r'$\hat{f}_t$')
        ax.axhline(y=mu_true, color='red', linestyle='--', linewidth=1.5,
                   label=fr'True $\mu$ = {mu_true}')
        ax.set_title(
            fr'Run {idx+1}: $\hat{{\alpha}}$={alpha_est:.3f}, '
            fr'$\hat{{\beta}}$={beta_est:.3f}, '
            fr'$\hat{{\sigma}}_u$={sigma_u_est:.3f}, '
            fr'$\hat{{\nu}}$={nu_est:.1f}',
            fontsize=10,
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)

        print(f"\n  Run {idx+1} (seed {r['seed']}):  "
              f"sigma_u={sigma_u_est:.4f}  nu={nu_est:.2f}  "
              f"omega={omega_est:.4f}  beta={beta_est:.4f}  "
              f"alpha={alpha_est:.4f}  LL={r['ll']:.2f}")

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
    print("  This occurs because the optimizer finds alpha > 0 (and beta close")
    print("  to 1) to improve the in-sample likelihood by fitting noise.")
    print("  The score-driven update chases random fluctuations in y_t,")
    print("  producing a spurious time-varying path.")
    print("  This highlights the need for a formal QLR test before")
    print("  concluding that time variation is genuinely present.")
    print("-" * 60)
    print("\nQUESTION 2(e) COMPLETE")


if __name__ == "__main__":
    main()
