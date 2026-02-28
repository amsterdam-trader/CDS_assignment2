# -*- coding: utf-8 -*-

# Question 1b
import numpy as np
import pandas as pd 
from pathlib import Path
from numpy.linalg import slogdet
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
import matplotlib.pyplot as plt

# set the path
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q1"
FIG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------------------
# Weight matrix construction
# --------------------------------------------------------------------------------------
def make_W_base(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construct the initial W (before normalization) exactly as in the assignment:
    (1) iid U[0,1]
    (2) abs value
    (3) diagonal set to zero
    """
    W = rng.random((n, n))           # U[0,1]
    W = np.abs(W)                    # abs (redundant but included)
    np.fill_diagonal(W, 0.0)         # no self-spillovers
    return W


def row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-normalize W: each row sums to 1 (when row sum > 0)."""
    row_sums = W.sum(axis=1, keepdims=True)
    # Avoid division by zero (should not occur here, but safe)
    row_sums[row_sums == 0.0] = 1.0
    return W / row_sums


def spectral_radius_power(W: np.ndarray, iters: int = 200, tol: float = 1e-10) -> float:
    """
    Approximate the spectral radius (largest absolute eigenvalue) using power iteration.

    This is much faster than a full eigenvalue decomposition for n=1000/2000.
    W is nonnegative here, so Perron-Frobenius implies the largest eigenvalue is real.
    """
    n = W.shape[0]
    v = np.ones(n)
    v = v / np.linalg.norm(v)

    lam_old = 0.0
    for _ in range(iters):
        wv = W @ v
        norm_wv = np.linalg.norm(wv)
        if norm_wv == 0.0:
            return 0.0
        v = wv / norm_wv
        lam = float(v @ (W @ v))  # Rayleigh quotient
        if abs(lam - lam_old) < tol * (1.0 + abs(lam_old)):
            break
        lam_old = lam

    return abs(lam)


def spectral_normalize(W: np.ndarray) -> np.ndarray:
    """Normalize W by dividing by its spectral radius."""
    sr = spectral_radius_power(W)
    if sr == 0.0:
        raise ValueError("Spectral radius estimated as 0; cannot normalize.")
    return W / sr


# --------------------------------------------------------------------------------------
# Data generation for SAR without regressors
# --------------------------------------------------------------------------------------
def simulate_y(W: np.ndarray, rho: float, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate y from:
        y = rho W y + e,  e ~ N(0, sigma2 I),
    i.e. (I - rho W) y = e.
    """
    n = W.shape[0]
    I = np.eye(n)
    e = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=(n, 1))
    A = I - rho * W
    y = solve(A, e, assume_a="gen")  # solve linear system
    return y


# --------------------------------------------------------------------------------------
# SAR estimation without regressors (same logic as Q1a)
# --------------------------------------------------------------------------------------
def concentrated_loglik_noX(rho: float, y: np.ndarray, W: np.ndarray) -> float:
    """
    Concentrated log-likelihood (up to constants):
        L(rho) = - (n/2) log(sigma2_hat(rho)) + log det(I - rho W),
    where sigma2_hat(rho) = (1/n) || (I - rho W) y ||^2.
    """
    n = y.shape[0]
    I = np.eye(n)
    S = I - rho * W

    sign, logdet = slogdet(S)
    if sign <= 0:
        return -np.inf

    Sy = S @ y
    sigma2_hat = ((Sy.T @ Sy) / n).item()
    if sigma2_hat <= 0.0 or (not np.isfinite(sigma2_hat)):
        return -np.inf

    return -0.5 * n * np.log(sigma2_hat) + float(logdet)


def estimate_rho_noX(y: np.ndarray, W: np.ndarray, bounds: tuple[float, float] = (-0.99, 0.99)) -> float:
    """Estimate rho by maximizing the concentrated log-likelihood over bounds."""
    obj = lambda r: -concentrated_loglik_noX(r, y, W)
    res = minimize_scalar(obj, bounds=bounds, method="bounded")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return float(res.x)


def estimate_sigma2_given_rho(y: np.ndarray, W: np.ndarray, rho: float) -> float:
    """Compute sigma2_hat(rho) = (1/n) ||(I - rho W) y||^2."""
    n = y.shape[0]
    I = np.eye(n)
    S = I - rho * W
    Sy = S @ y
    return ((Sy.T @ Sy) / n).item()


def fit_sar_noX(y: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """
    Fit SAR without regressors:
    returns (rho_hat, sigma2_hat).
    """
    rho_hat = estimate_rho_noX(y, W)
    sigma2_hat = estimate_sigma2_given_rho(y, W, rho_hat)
    return rho_hat, sigma2_hat


# --------------------------------------------------------------------------------------
# Monte Carlo driver + plotting
# --------------------------------------------------------------------------------------
def run_mc(
    n: int,
    normalization: str,
    reps: int,
    rho_true: float = 0.5,
    sigma2_true: float = 1.0,
    seed: int = 123,
) -> dict:
    """
    Run Monte Carlo for given n and normalization type.
    normalization ∈ {"row", "spectral"}.
    Returns dict with arrays of rho_hats and sigma2_hats.
    """
    rng = np.random.default_rng(seed)

    rho_hats = np.empty(reps)
    sigma2_hats = np.empty(reps)

    for r in range(reps):
        # New W each replication (as in the assignment)
        W0 = make_W_base(n, rng)

        if normalization == "row":
            W = row_normalize(W0)
        elif normalization == "spectral":
            W = spectral_normalize(W0)
        else:
            raise ValueError("normalization must be 'row' or 'spectral'.")

        # Simulate y
        y = simulate_y(W, rho=rho_true, sigma2=sigma2_true, rng=rng)

        # Estimate parameters
        rho_hat, sigma2_hat = fit_sar_noX(y, W)
        rho_hats[r] = rho_hat
        sigma2_hats[r] = sigma2_hat

        if (r + 1) % max(1, reps // 10) == 0:
            print(f"[n={n}, norm={normalization}] finished {r+1}/{reps}")

    return {"rho_hat": rho_hats, "sigma2_hat": sigma2_hats}


def plot_hist(arr: np.ndarray, true_value: float, title: str, xlabel: str, outpath: Path) -> None:
    """Histogram normalized as a density, with a vertical line at the true value."""
    plt.figure()
    plt.hist(arr, bins=30, density=True, edgecolor="black")
    plt.axvline(true_value, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def bias_var_mse(x: np.ndarray, true: float) -> tuple[float, float, float]:
    """Overview table of the bias/variance (and MSE)."""
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    bias = mean - true
    mse = bias**2 + var
    return bias, var, mse


def main() -> None:
    # -----------------------------
    # User controls (edit as needed)
    # -----------------------------
    reps = 200              
    ns = [1000, 2000]
    normalizations = ["row", "spectral"]

    rho_true = 0.5
    sigma2_true = 1.0
    seed = 123

    # -----------------------------
    # Run experiments
    # -----------------------------
    all_results = {}
    rows = []

    for n in ns:
        for norm in normalizations:
            print(f"\n=== Running MC: n={n}, normalization={norm}, reps={reps} ===")
            res = run_mc(
                n=n,
                normalization=norm,
                reps=reps,
                rho_true=rho_true,
                sigma2_true=sigma2_true,
                seed=seed,
            )
            all_results[(n, norm)] = res

            b_rho, v_rho, mse_rho = bias_var_mse(res["rho_hat"], rho_true)
            b_s2, v_s2, mse_s2 = bias_var_mse(res["sigma2_hat"], sigma2_true)

            rows.append(("rho", norm, n, b_rho, v_rho, mse_rho))
            rows.append(("sigma2", norm, n, b_s2, v_s2, mse_s2))

            # Summary stats
            print(f"rho_hat:   mean={res['rho_hat'].mean():.4f}, std={res['rho_hat'].std(ddof=1):.4f}")
            print(f"sigma2_hat: mean={res['sigma2_hat'].mean():.4f}, std={res['sigma2_hat'].std(ddof=1):.4f}")

            # Plots
            plot_hist(
                res["rho_hat"],
                true_value=rho_true,
                title=f"Histogram of $\\hat{{\\rho}}$ (n={n}, norm={norm})",
                xlabel=r"$\hat{\rho}$",
                outpath=FIG_DIR / f"q1b_rhohat_n{n}_{norm}.png",
            )
            plot_hist(
                res["sigma2_hat"],
                true_value=sigma2_true,
                title=f"Histogram of $\\hat{{\\sigma}}^2$ (n={n}, norm={norm})",
                xlabel=r"$\hat{\sigma}^2$",
                outpath=FIG_DIR / f"q1b_sigma2hat_n{n}_{norm}.png",
            )

    print("\n Latex table of the bias/variance (and MSE):")
    for par, norm, n, b, v, mse in rows:
        label = r"$\hat{\rho}$" if par == "rho" else r"$\hat{\sigma}^2$"
        norm_label = "Row" if norm == "row" else "Spectral"
        print(f"{label} & {norm_label} & {n} & {b:+.4f} & {v:.4f} & {mse:.4f} \\\\")

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()