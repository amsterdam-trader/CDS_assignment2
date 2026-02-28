# -*- coding: utf-8 -*-

# Question 1d
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

data = pd.read_csv(ROOT / "data" / "iowa_yield_05_10.csv")

# --------------------------------------------------------------------------------------
# Construction Weight Matrices
# --------------------------------------------------------------------------------------
def make_W_dense(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construct the dense W (before normalization) exactly as in the previous subquestion (q1_b)
    """
    W = rng.random((n, n))           # U[0,1]
    W = np.abs(W)                    # abs (redundant but included)
    np.fill_diagonal(W, 0.0)         # no self-spillovers
    return W

def make_W_sparse(n:int) -> np.ndarray:
    """
    Construct the sparse W (before normalization) by having each unit be connected only to its left and right neighbours*
    *Except on the first row, where it's only the right neighbour, and the last row, where it's only the left neighbour
    """
    W = np.zeros((n,n))
    
    for i in range(n):
        if i > 0:
            W[i, i-1] = 1
        if i < n-1:
            W[i, i+1] = 1
    
    return W

def row_normalize(W: np.ndarray) -> np.ndarray:
    """Row-normalize W"""
    row_sums = W.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0.0] = 1.0
    return W / row_sums

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
# SAR estimation without regressors (same logic as q1_a and q1_b)
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

def run_mc(n: int, Wtype: str, reps: int, rho_true: float = 0.5, sigma2_true: float = 1.0, seed: int = 123) -> dict:
    """
    Run Monte Carlo for given n.
    Returns dict with arrays of rho_hats and sigma2_hats.
    """
    rng = np.random.default_rng(seed)

    rho_hats = np.empty(reps)
    sigma2_hats = np.empty(reps)

    for r in range(reps):
        # New W each replication (as in the assignment)
        if Wtype == "dense":
            W0 = make_W_dense(n, rng)
            W = row_normalize(W0)
        elif Wtype == "sparse":
            W0 = make_W_sparse(n)
            W = row_normalize(W0)
        else:
            raise ValueError("Wtype must be 'dense' or 'sparse'.")

        # Simulate y
        y = simulate_y(W, rho=rho_true, sigma2=sigma2_true, rng=rng)

        # Estimate parameters
        rho_hat, sigma2_hat = fit_sar_noX(y, W)
        rho_hats[r] = rho_hat
        sigma2_hats[r] = sigma2_hat

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

def main() -> None:
    # -----------------------------
    # User controls (edit as needed)
    # -----------------------------
    reps = 200 
    ns = [500, 1000]             
    Wtypes = ["dense", "sparse"]

    rho_true = 0.5
    sigma2_true = 1.0
    seed = 123

    # -----------------------------
    # Run experiments
    # -----------------------------
    all_results = {}

    for n in ns:
        for type in Wtypes:
            print(f"\n=== Running MC: n={n}, Wtype={type}, reps={reps} ===")
            res = run_mc(
                n=n,
                Wtype=type,
                reps=reps,
                rho_true=rho_true,
                sigma2_true=sigma2_true,
                seed=seed,
            )
            all_results[(n, type)] = res

            # Summary stats
            print(f"rho_hat:   mean={res['rho_hat'].mean():.4f}, std={res['rho_hat'].std(ddof=1):.4f}")
            print(f"sigma2_hat: mean={res['sigma2_hat'].mean():.4f}, std={res['sigma2_hat'].std(ddof=1):.4f}")

            # Plots
            plot_hist(
                res["rho_hat"],
                true_value=rho_true,
                title=f"Histogram of $\\hat{{\\rho}}$ (n={n}, Wtype={type})",
                xlabel=r"$\hat{\rho}$",
                outpath=FIG_DIR / f"q1d_rhohat_n{n}_{type}.png",
            )
            plot_hist(
                res["sigma2_hat"],
                true_value=sigma2_true,
                title=f"Histogram of $\\hat{{\\sigma}}^2$ (n={n}, Wtype={type})",
                xlabel=r"$\hat{\sigma}^2$",
                outpath=FIG_DIR / f"q1d_sigma2hat_n{n}_{type}.png",
            )

    print("\nDone. Figures saved to:", FIG_DIR)

if __name__ == "__main__":
    main()