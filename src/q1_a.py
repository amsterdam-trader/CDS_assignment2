# -*- coding: utf-8 -*-

# Question 1a
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import slogdet
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

# set the path
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q1"
FIG_DIR.mkdir(exist_ok=True)

data_yield = pd.read_csv(ROOT / "data" / "iowa_yield_05_10.csv")
data_normalized = pd.read_csv(ROOT / "data" / "iowa_spatial_weight_row_normalized.csv")

# --------------------------------------------------------------------------------------
# Data structuring 
# --------------------------------------------------------------------------------------
def prepare_inputs_from_loaded_dataframes(
    yield_df: pd.DataFrame,
    w_df: pd.DataFrame
):
    """
    Convert the already-loaded DataFrames into the numeric objects used by the SAR code.

    Expected yield_df columns: [county_name, yield_2005, yield_2010]

    The weight matrix file may include county names in the first column and/or header row.
    This function robustly extracts the numeric W.

    Returns:
        county_names : list[str]
        y_2005 : (n, 1) ndarray
        y_2010 : (n, 1) ndarray
        W      : (n, n) ndarray (float)
    """
    county_names = yield_df.iloc[:, 0].astype(str).tolist()
    y_2005 = yield_df.iloc[:, 1].to_numpy(dtype=float).reshape(-1, 1)
    y_2010 = yield_df.iloc[:, 2].to_numpy(dtype=float).reshape(-1, 1)

    n = y_2010.shape[0]

    # --- Robust extraction of numeric W ---
    w_work = w_df.copy()

    # If the first column is non-numeric (e.g., county names), drop it.
    if not pd.api.types.is_numeric_dtype(w_work.iloc[:, 0]):
        w_work = w_work.iloc[:, 1:]

    # If column names are county names (strings), that's fine; we only need the values.
    # Coerce everything to numeric, turning any leftover strings into NaN.
    w_work = w_work.apply(pd.to_numeric, errors="coerce")

    # Drop any columns that became all-NaN (can happen if there is an empty label column)
    w_work = w_work.dropna(axis=1, how="all")

    # Now check rows: if we still have NaNs due to a header-like row inside data, drop such rows.
    # (We expect a full numeric matrix.)
    w_work = w_work.dropna(axis=0, how="any")

    W = w_work.to_numpy(dtype=float)

    if W.shape != (n, n):
        raise ValueError(
            f"After cleaning, W has shape {W.shape} but expected {(n, n)}. "
            "This likely means the weight matrix file includes extra label rows/cols."
        )

    # Sanity checks
    diag_max = float(np.max(np.abs(np.diag(W))))
    if diag_max > 1e-10:
        print(f"Warning: max |diag(W)| = {diag_max:.3e} (expected ~0).")

    row_sums = W.sum(axis=1)
    max_row_sum_dev = float(np.max(np.abs(row_sums - 1.0)))
    if max_row_sum_dev > 1e-6:
        print(f"Warning: max deviation of row sums from 1 is {max_row_sum_dev:.3e} (expected ~1).")

    return county_names, y_2005, y_2010, W

# --------------------------------------------------------------------------------------
# SAR likelihood / estimation / LR test (no regressors)
# --------------------------------------------------------------------------------------
def concentrated_loglik_noX(rho: float, y: np.ndarray, W: np.ndarray) -> float:
    """
    Concentrated log-likelihood (up to constants) for SAR without regressors.

    L(ρ) = - (n/2) log(σ̂^2(ρ)) + log det(S(ρ)),
    where S(ρ) = I - ρW and σ̂^2(ρ) = (1/n) ||S(ρ)y||^2.

    Returns -inf if det(S(ρ)) <= 0 or σ̂^2(ρ) is invalid.
    """
    n = y.shape[0]
    I = np.eye(n)

    S = I - rho * W
    sign, logdet = slogdet(S)
    if sign <= 0:
        return -np.inf

    Sy = S @ y
    sigma2_hat = ((Sy.T @ Sy) / n).item()
    if sigma2_hat <= 0 or (not np.isfinite(sigma2_hat)):
        return -np.inf

    return -0.5 * n * np.log(sigma2_hat) + float(logdet)


def estimate_rho_noX(
    y: np.ndarray,
    W: np.ndarray,
    rho_bounds: tuple[float, float] = (-0.99, 0.99),
) -> float:
    """
    Estimate rho by maximizing the concentrated log-likelihood for the no-regressor SAR.
    """
    obj = lambda r: -concentrated_loglik_noX(r, y, W)
    res = minimize_scalar(obj, bounds=rho_bounds, method="bounded")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return float(res.x)


def compute_lr_test_noX(
    y: np.ndarray,
    W: np.ndarray,
    rho_bounds: tuple[float, float] = (-0.99, 0.99),
) -> dict:
    """
    Compute LR test statistic for H0: rho=0 in the no-regressor SAR model.
    Returns a dict with rho_hat, LR, p_value, and intermediate quantities.
    """
    n = y.shape[0]
    I = np.eye(n)

    rho_hat = estimate_rho_noX(y, W, rho_bounds=rho_bounds)
    S_hat = I - rho_hat * W
    sign, logdet_S_hat = slogdet(S_hat)
    if sign <= 0:
        raise RuntimeError("Estimated rho produces non-positive determinant for S(rho_hat).")

    sigma2_hat_0 = ((y.T @ y) / n).item()  # restricted: rho=0 
    Sy_hat = S_hat @ y
    sigma2_hat_1 = ((Sy_hat.T @ Sy_hat) / n).item()  # unrestricted at rho_hat

    LR = n * (np.log(sigma2_hat_0) - np.log(sigma2_hat_1)) + 2.0 * float(logdet_S_hat)
    p_value = chi2.sf(LR, df=1)  # sf = 1 - cdf, more stable for tiny p-values

    return {
        "rho_hat": rho_hat,
        "sigma2_hat_0": sigma2_hat_0,
        "sigma2_hat_1": sigma2_hat_1,
        "logdet_S_hat": float(logdet_S_hat),
        "LR": float(LR),
        "p_value": float(p_value),
    }


def print_results(label: str, res: dict) -> None:
    """
    Print results + χ²(1) critical values, and a 5% decision.
    """
    print(f"\n=== {label} ===")
    print(f"rho_hat        = {res['rho_hat']:.6f}")
    print(f"sigma2_hat_0   = {res['sigma2_hat_0']:.6f}   (restricted: rho=0)")
    print(f"sigma2_hat_1   = {res['sigma2_hat_1']:.6f}   (unrestricted: rho=rho_hat)")
    print(f"logdet(S_hat)  = {res['logdet_S_hat']:.6f}")
    print(f"LR statistic   = {res['LR']:.6f}")
    print(f"p-value (χ²(1))= {res['p_value']:.3e}")
    
    cv_10 = chi2.ppf(0.90, df=1)
    cv_05 = chi2.ppf(0.95, df=1)
    cv_01 = chi2.ppf(0.99, df=1)
    print(f"Critical values χ²(1): 10%={cv_10:.4f}, 5%={cv_05:.4f}, 1%={cv_01:.4f}")

    if res["LR"] >= cv_05:
        print("Decision at 5%: Reject H0 (evidence of spatial autoregression).")
    else:
        print("Decision at 5%: Do NOT reject H0 (no strong evidence of spatial autoregression).")


# --------------------------------------------------------------------------------------
# Run for 2010 and 2005
# --------------------------------------------------------------------------------------
def main() -> None:
    """
    - Use 2010 yields to replicate the slide example (ρ̂ close to 1, large LR).
    - Repeat for 2005 and compare conclusions.
    """
    _, y_2005, y_2010, W = prepare_inputs_from_loaded_dataframes(
        data_yield, data_normalized
    )

    res_2010 = compute_lr_test_noX(y_2010, W)
    print_results("2010 yields", res_2010)

    res_2005 = compute_lr_test_noX(y_2005, W)
    print_results("2005 yields", res_2005)

    print("\n=== Comparison (quick note) ===")
    print(f"2010: rho_hat={res_2010['rho_hat']:.6f}, LR={res_2010['LR']:.6f}")
    print(f"2005: rho_hat={res_2005['rho_hat']:.6f}, LR={res_2005['LR']:.6f}")
    print("Interpretation: compare LR to χ²(1) critical values; large LR => strong spatial dependence.")


if __name__ == "__main__":
    main()