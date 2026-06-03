"""
Process Data Modeling Lab - core numerical routines.

This module is intentionally dependency-light: NumPy and pandas are enough for
all model calculations, while matplotlib is only used by the runner/notebook.
The routines are written explicitly so that the numerical steps can be inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd


EPS = 1.0e-12


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def _as_2d(X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be a 1D or 2D numeric array")
    return arr


def _as_1d(y: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError("y must be a 1D numeric array")
    return arr


def safe_std(x: np.ndarray, axis: int = 0, ddof: int = 1) -> np.ndarray:
    s = np.nanstd(x, axis=axis, ddof=ddof)
    if np.isscalar(s):
        return np.array(1.0 if abs(float(s)) < EPS else float(s))
    s = np.asarray(s, dtype=float)
    s[~np.isfinite(s)] = 1.0
    s[np.abs(s) < EPS] = 1.0
    return s


def standardize_fit(X: np.ndarray, ddof: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _as_2d(X)
    mean = np.nanmean(X, axis=0)
    std = safe_std(X, axis=0, ddof=ddof)
    return (X - mean) / std, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = _as_2d(X)
    return (X - mean) / std


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = _as_2d(X)
    return np.column_stack([np.ones(X.shape[0]), X])


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    return float(np.nanmean(np.abs(y_true - y_pred)))


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + EPS))


def metrics_table(y_true: Sequence[float], predictions: Dict[str, Sequence[float]]) -> pd.DataFrame:
    rows = []
    for name, pred in predictions.items():
        rows.append({"model": name, "RMSE": rmse(y_true, pred), "MAE": mae(y_true, pred), "R2": r2_score(y_true, pred)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def train_test_split_time(X: np.ndarray, y: np.ndarray, train_fraction: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = _as_2d(X)
    y = _as_1d(y)
    if len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    cut = int(round(X.shape[0] * train_fraction))
    cut = min(max(cut, 1), X.shape[0] - 1)
    return X[:cut], X[cut:], y[:cut], y[cut:]


# ---------------------------------------------------------------------------
# Linear models: OLS, Ridge, PCA/PCR, PLS1
# ---------------------------------------------------------------------------

@dataclass
class LinearModel:
    coef: np.ndarray
    intercept: float
    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    y_mean: Optional[float] = None
    y_std: Optional[float] = None
    coef_std: Optional[np.ndarray] = None
    name: str = "linear"

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        return X @ self.coef + self.intercept


def ols_fit(X: np.ndarray, y: np.ndarray, standardize: bool = False) -> LinearModel:
    X = _as_2d(X)
    y = _as_1d(y)
    if standardize:
        Xs, x_mean, x_std = standardize_fit(X)
        y_mean = float(np.mean(y))
        y_std = float(safe_std(y, axis=0, ddof=1))
        ys = (y - y_mean) / y_std
        beta_std, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        coef = beta_std * y_std / x_std
        intercept = y_mean - float(x_mean @ coef)
        return LinearModel(coef=coef, intercept=intercept, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, coef_std=beta_std, name="OLS standardized")
    beta, *_ = np.linalg.lstsq(add_intercept(X), y, rcond=None)
    return LinearModel(coef=beta[1:], intercept=float(beta[0]), name="OLS")


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, standardize: bool = True) -> LinearModel:
    X = _as_2d(X)
    y = _as_1d(y)
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if standardize:
        Xs, x_mean, x_std = standardize_fit(X)
        y_mean = float(np.mean(y))
        y_std = float(safe_std(y, axis=0, ddof=1))
        ys = (y - y_mean) / y_std
        p = Xs.shape[1]
        beta_std = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(p), Xs.T @ ys)
        coef = beta_std * y_std / x_std
        intercept = y_mean - float(x_mean @ coef)
        return LinearModel(coef=coef, intercept=intercept, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, coef_std=beta_std, name=f"Ridge alpha={alpha:g}")
    X1 = add_intercept(X)
    p = X1.shape[1]
    penalty = alpha * np.eye(p)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)
    return LinearModel(coef=beta[1:], intercept=float(beta[0]), name=f"Ridge alpha={alpha:g}")


@dataclass
class PCRModel:
    coef: np.ndarray
    intercept: float
    components: np.ndarray
    singular_values: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    n_components: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        return X @ self.coef + self.intercept


def pcr_fit(X: np.ndarray, y: np.ndarray, n_components: int) -> PCRModel:
    X = _as_2d(X)
    y = _as_1d(y)
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    max_components = min(X.shape)
    n_components = min(n_components, max_components)
    Xs, x_mean, x_std = standardize_fit(X)
    y_mean = float(np.mean(y))
    y_std = float(safe_std(y, axis=0, ddof=1))
    ys = (y - y_mean) / y_std
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    V = Vt.T[:, :n_components]
    T = Xs @ V
    gamma, *_ = np.linalg.lstsq(T, ys, rcond=None)
    beta_std = V @ gamma
    coef = beta_std * y_std / x_std
    intercept = y_mean - float(x_mean @ coef)
    return PCRModel(coef=coef, intercept=intercept, components=V, singular_values=S, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, n_components=n_components)


@dataclass
class PLSModel:
    coef: np.ndarray
    intercept: float
    coef_std: np.ndarray
    W: np.ndarray
    P: np.ndarray
    d: np.ndarray
    T: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    n_components: int
    x_explained_per_component: np.ndarray
    y_explained_per_component: np.ndarray
    sample_weight: Optional[np.ndarray] = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        return X @ self.coef + self.intercept


def _weighted_mean_std(X: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.clip(w, 0.0, np.inf)
    w = w / (np.sum(w) + EPS)
    mu = np.sum(X * w[:, None], axis=0)
    var = np.sum(w[:, None] * (X - mu) ** 2, axis=0)
    std = np.sqrt(np.maximum(var, EPS))
    std[std < EPS] = 1.0
    return mu, std


def pls1_fit(X: np.ndarray, y: np.ndarray, n_components: int = 2, sample_weight: Optional[np.ndarray] = None) -> PLSModel:
    """Fit PLS1 with a transparent NIPALS-style algorithm.

    The returned model predicts in original units. X and y are standardized inside
    the routine. When sample_weight is provided, weighted centering/scaling and a
    sqrt(weight) transformation are used.
    """
    X = _as_2d(X)
    y = _as_1d(y)
    if len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    n, p = X.shape
    max_components = min(n - 1, p)
    n_components = int(min(n_components, max_components))
    if n_components < 1:
        raise ValueError("not enough samples/features for PLS")

    if sample_weight is None:
        x_mean = np.mean(X, axis=0)
        x_std = safe_std(X, axis=0, ddof=1)
        y_mean = float(np.mean(y))
        y_std = float(safe_std(y, axis=0, ddof=1))
        Xs = (X - x_mean) / x_std
        ys = (y - y_mean) / y_std
        weights_used = None
        X_algorithm = Xs.copy()
        y_algorithm = ys.copy()
    else:
        w0 = np.asarray(sample_weight, dtype=float).reshape(-1)
        if w0.shape[0] != n:
            raise ValueError("sample_weight length must match number of rows")
        w0 = np.clip(w0, 0.0, np.inf)
        if np.sum(w0) < EPS:
            w0 = np.ones(n)
        w0 = w0 / np.mean(w0)
        x_mean, x_std = _weighted_mean_std(X, w0)
        y_mean_arr, y_std_arr = _weighted_mean_std(y.reshape(-1, 1), w0)
        y_mean = float(y_mean_arr[0])
        y_std = float(y_std_arr[0])
        Xs = (X - x_mean) / x_std
        ys = (y - y_mean) / y_std
        sqrt_w = np.sqrt(w0)
        X_algorithm = Xs * sqrt_w[:, None]
        y_algorithm = ys * sqrt_w
        weights_used = w0

    Xr = X_algorithm.copy()
    yr = y_algorithm.copy()
    total_x_ss = float(np.sum(X_algorithm ** 2)) + EPS
    total_y_ss = float(np.sum(y_algorithm ** 2)) + EPS

    W_list, P_list, d_list, T_list = [], [], [], []
    x_exp, y_exp = [], []

    for _ in range(n_components):
        w = Xr.T @ yr
        w_norm = np.linalg.norm(w)
        if w_norm < EPS:
            break
        w = w / w_norm
        t = Xr @ w
        tt = float(t.T @ t)
        if tt < EPS:
            break
        p_vec = (Xr.T @ t) / tt
        d_val = float(yr.T @ t / tt)
        X_hat = np.outer(t, p_vec)
        y_hat = d_val * t
        Xr = Xr - X_hat
        yr = yr - y_hat
        W_list.append(w)
        P_list.append(p_vec)
        d_list.append(d_val)
        T_list.append(t)
        x_exp.append(float(np.sum(X_hat ** 2) / total_x_ss))
        y_exp.append(float(np.sum(y_hat ** 2) / total_y_ss))

    if not W_list:
        # Fallback to a tiny ridge model if all covariance vanishes.
        ridge = ridge_fit(X, y, alpha=1.0, standardize=True)
        return PLSModel(coef=ridge.coef, intercept=ridge.intercept, coef_std=np.asarray(ridge.coef_std), W=np.zeros((p, 0)), P=np.zeros((p, 0)), d=np.zeros(0), T=np.zeros((n, 0)), x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, n_components=0, x_explained_per_component=np.zeros(0), y_explained_per_component=np.zeros(0), sample_weight=weights_used)

    W = np.column_stack(W_list)
    P = np.column_stack(P_list)
    d = np.asarray(d_list)
    T = np.column_stack(T_list)
    # Regression coefficient in standardized original Xs -> standardized y.
    beta_std = W @ np.linalg.pinv(P.T @ W) @ d
    coef = beta_std * y_std / x_std
    intercept = y_mean - float(x_mean @ coef)
    return PLSModel(coef=coef, intercept=intercept, coef_std=beta_std, W=W, P=P, d=d, T=T, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, n_components=W.shape[1], x_explained_per_component=np.asarray(x_exp), y_explained_per_component=np.asarray(y_exp), sample_weight=weights_used)


def kfold_indices(n: int, k: int = 5, shuffle: bool = True, seed: int = 0) -> List[np.ndarray]:
    if k < 2:
        raise ValueError("k must be >= 2")
    k = min(k, n)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    return [fold.astype(int) for fold in np.array_split(idx, k)]


def pls_cross_validate(X: np.ndarray, y: np.ndarray, max_components: int = 8, k: int = 5, seed: int = 0, shuffle: bool = True) -> pd.DataFrame:
    X = _as_2d(X)
    y = _as_1d(y)
    max_components = min(max_components, X.shape[1], X.shape[0] - 1)
    folds = kfold_indices(X.shape[0], k=k, shuffle=shuffle, seed=seed)
    rows = []
    for r in range(1, max_components + 1):
        fold_errors = []
        for valid_idx in folds:
            train_idx = np.setdiff1d(np.arange(X.shape[0]), valid_idx)
            model = pls1_fit(X[train_idx], y[train_idx], n_components=r)
            pred = model.predict(X[valid_idx])
            fold_errors.append(float(np.sum((y[valid_idx] - pred) ** 2)))
        press = float(np.sum(fold_errors))
        rows.append({"components": r, "PRESS": press, "CV_RMSE": math.sqrt(press / X.shape[0])})
    return pd.DataFrame(rows)


def pcr_cross_validate(X: np.ndarray, y: np.ndarray, max_components: int = 8, k: int = 5, seed: int = 0, shuffle: bool = True) -> pd.DataFrame:
    X = _as_2d(X)
    y = _as_1d(y)
    max_components = min(max_components, X.shape[1], X.shape[0] - 1)
    folds = kfold_indices(X.shape[0], k=k, shuffle=shuffle, seed=seed)
    rows = []
    for r in range(1, max_components + 1):
        fold_errors = []
        for valid_idx in folds:
            train_idx = np.setdiff1d(np.arange(X.shape[0]), valid_idx)
            model = pcr_fit(X[train_idx], y[train_idx], n_components=r)
            pred = model.predict(X[valid_idx])
            fold_errors.append(float(np.sum((y[valid_idx] - pred) ** 2)))
        press = float(np.sum(fold_errors))
        rows.append({"components": r, "PRESS": press, "CV_RMSE": math.sqrt(press / X.shape[0])})
    return pd.DataFrame(rows)


def vif_table(X: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    X = _as_2d(X)
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(X.shape[1])]
    rows = []
    for j in range(X.shape[1]):
        yj = X[:, j]
        X_other = np.delete(X, j, axis=1)
        if X_other.shape[1] == 0:
            vif = 1.0
        else:
            mdl = ols_fit(X_other, yj, standardize=False)
            pred = mdl.predict(X_other)
            r2 = r2_score(yj, pred)
            vif = float(1.0 / max(EPS, 1.0 - r2))
        rows.append({"feature": feature_names[j], "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def condition_number(X: np.ndarray, standardize: bool = True) -> float:
    X = _as_2d(X)
    X_use = standardize_fit(X)[0] if standardize else X.copy()
    s = np.linalg.svd(X_use, compute_uv=False)
    return float(np.max(s) / (np.min(s) + EPS))


# ---------------------------------------------------------------------------
# Synthetic process data generators
# ---------------------------------------------------------------------------

def make_simple_regression(seed: int = 1, n: int = 50, corr_strength: float = 0.72) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = corr_strength * x + math.sqrt(max(0.0, 1.0 - corr_strength**2)) * z
    return x.reshape(-1, 1), y


def make_multicollinearity_data(seed: int = 2, n: int = 80, noise_x: float = 0.05, noise_y: float = 0.3) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    base1 = rng.uniform(1.0, 9.0, size=n)
    base2 = rng.uniform(1.0, 9.0, size=n)
    # Near-constraint: x1 + x2 + x3 is almost constant.
    x1 = base1 + rng.normal(scale=noise_x, size=n)
    x2 = base2 + rng.normal(scale=noise_x, size=n)
    x3 = 14.0 - base1 - base2 + rng.normal(scale=noise_x, size=n)
    X = np.column_stack([x1, x2, x3])
    true_coef = np.array([0.25, -0.75, 0.35])
    y = X @ true_coef + rng.normal(scale=noise_y, size=n)
    names = ["x1", "x2", "x3"]
    return X, y, names, true_coef


def make_soft_sensor_process(seed: int = 3, n: int = 420, maintenance_period: int = 95, noise: float = 0.25) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    age = t % maintenance_period
    cycle = 2.0 * np.pi * t / 60.0
    production_mode = ((t // 120) % 2).astype(float)

    temperature = 65 + 0.04 * age + 1.8 * np.sin(cycle) + rng.normal(scale=0.6, size=n)
    pressure = 1.2 + 0.004 * age + 0.12 * production_mode + rng.normal(scale=0.03, size=n)
    flow_a = 100 + 0.18 * age - 4.5 * production_mode + rng.normal(scale=1.2, size=n)
    flow_b = 55 + 0.5 * flow_a / 100 + 1.0 * np.cos(cycle / 2) + rng.normal(scale=0.5, size=n)
    power = 250 + 0.32 * age + 4 * np.sin(cycle / 3) + rng.normal(scale=2.5, size=n)
    exhaust = temperature - 4.5 + 0.02 * flow_a + rng.normal(scale=0.5, size=n)
    vibration = 0.2 + 0.003 * age + rng.normal(scale=0.02, size=n)
    catalyst_signal = np.exp(-age / 160.0) + 0.02 * rng.normal(size=n)
    maintenance_flag = (age == 0).astype(float)
    humidity = 45 + 6 * np.sin(2.0 * np.pi * t / 180.0) + rng.normal(scale=1.3, size=n)
    ambient = 23 + 3.0 * np.sin(2.0 * np.pi * t / 365.0) + rng.normal(scale=0.7, size=n)
    raw_sensor = power + 8 * pressure + rng.normal(scale=1.0, size=n)

    # A nonlinear but smooth quality relationship with drift/reset behavior.
    y = (
        0.055 * temperature
        - 1.65 * pressure
        + 0.018 * flow_a
        + 0.012 * power
        - 0.35 * production_mode
        + 0.006 * age
        # Nonlinear local behavior: after maintenance and in different modes,
        # the relationship is not well represented by one global linear model.
        + 0.85 * np.sin(age / 11.0)
        + 0.85 * production_mode * np.sin((flow_a - 95) / 7.0)
        + 0.45 * np.sin((flow_a - 95) / 13)
        - 0.25 * np.square(vibration * 4)
        + 0.18 * catalyst_signal
        + rng.normal(scale=noise, size=n)
    )
    df = pd.DataFrame(
        {
            "temperature": temperature,
            "pressure": pressure,
            "flow_a": flow_a,
            "flow_b": flow_b,
            "power": power,
            "exhaust": exhaust,
            "vibration": vibration,
            "catalyst_signal": catalyst_signal,
            "maintenance_flag": maintenance_flag,
            "humidity": humidity,
            "ambient": ambient,
            "raw_sensor": raw_sensor,
            "age_since_maintenance": age,
            "production_mode": production_mode,
        }
    )
    return df, pd.Series(y, name="quality")


def make_graybox_process(seed: int = 4, n: int = 260, noise: float = 0.18) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    age = t % 75
    feed = rng.uniform(0.7, 1.5, size=n)
    temp_c = 55 + 12 * rng.random(n) + 0.04 * age
    pressure = 1.0 + 0.25 * rng.random(n) + 0.002 * age
    residence = 2.5 + 0.8 * rng.random(n)
    impurity = 0.05 + 0.05 * rng.random(n) + 0.0008 * age
    # Physics skeleton known to an engineer.
    g = feed * np.exp(-850.0 / (temp_c + 273.15)) * residence / pressure
    theta_true = 12.0 + 1.8 * np.sin(temp_c / 9.0) - 4.0 * impurity - 0.011 * age
    residual = 0.45 * np.sin(4.2 * feed) + 0.35 * (pressure - 1.1) ** 2 + 0.04 * (temp_c - 61)
    y = theta_true * g + residual + rng.normal(scale=noise, size=n)
    X = pd.DataFrame(
        {
            "feed": feed,
            "temperature_c": temp_c,
            "pressure": pressure,
            "residence_time": residence,
            "impurity": impurity,
            "age_since_cleaning": age,
            "physics_signal_g": g,
        }
    )
    return X, pd.Series(y, name="quality"), pd.Series(theta_true, name="theta_true")


def make_transfer_domains(seed: int = 5, n_source: int = 220, n_target_train: int = 12, n_target_test: int = 100, n_common: int = 8, n_source_unique: int = 4, n_target_unique: int = 4, noise: float = 0.35) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta_c = rng.normal(scale=0.8, size=n_common)
    beta_us = rng.normal(scale=0.35, size=n_source_unique)
    beta_ut = rng.normal(scale=0.55, size=n_target_unique)
    delta_t = rng.normal(scale=0.15, size=n_common)

    def common(n: int, shift: float) -> np.ndarray:
        z = rng.normal(size=(n, n_common))
        # Correlate adjacent variables.
        for j in range(1, n_common):
            z[:, j] = 0.65 * z[:, j - 1] + 0.35 * z[:, j]
        return z + shift

    Cs = common(n_source, 0.0)
    Us = rng.normal(size=(n_source, n_source_unique))
    Ct_all = common(n_target_train + n_target_test, 0.35)
    Ut_all = rng.normal(size=(n_target_train + n_target_test, n_target_unique))
    yt_all = Ct_all @ (beta_c + delta_t) + Ut_all @ beta_ut + 0.3 * np.sin(Ct_all[:, 0]) + rng.normal(scale=noise, size=n_target_train + n_target_test)
    ys = Cs @ beta_c + Us @ beta_us + 0.3 * np.sin(Cs[:, 0]) + rng.normal(scale=noise, size=n_source)
    return {
        "Cs": Cs,
        "Us": Us,
        "ys": ys,
        "Ct_train": Ct_all[:n_target_train],
        "Ut_train": Ut_all[:n_target_train],
        "yt_train": yt_all[:n_target_train],
        "Ct_test": Ct_all[n_target_train:],
        "Ut_test": Ut_all[n_target_train:],
        "yt_test": yt_all[n_target_train:],
        "beta_c": beta_c,
        "beta_us": beta_us,
        "beta_ut": beta_ut,
    }


def make_raw_data_diagnostic_case(seed: int = 6, n: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    normal = rng.normal(0, 1, size=n)
    outlier = rng.normal(0, 1, size=n)
    outlier[[45, 180, 290]] += np.array([8.0, -7.0, 6.5])
    range_shift = rng.normal(0, 0.6, size=n)
    range_shift[160:] += 3.0
    trend = 0.012 * t + rng.normal(0, 0.35, size=n)
    periodic = 1.8 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 0.25, size=n)
    missing = rng.normal(0, 1, size=n)
    missing[120:150] = np.nan
    lower_clip = rng.normal(0.4, 0.7, size=n)
    lower_clip = np.maximum(lower_clip, -0.25)
    bimodal = np.r_[rng.normal(-1.4, 0.35, size=n // 2), rng.normal(1.4, 0.35, size=n - n // 2)]
    rng.shuffle(bimodal)
    correlated = 0.9 * normal + rng.normal(0, 0.12, size=n)
    return pd.DataFrame(
        {
            "normal": normal,
            "outlier_like": outlier,
            "range_shift": range_shift,
            "trend": trend,
            "periodic": periodic,
            "missing_block": missing,
            "lower_clipped": lower_clip,
            "bimodal": bimodal,
            "correlated_with_normal": correlated,
        }
    )


# ---------------------------------------------------------------------------
# JIT / local models
# ---------------------------------------------------------------------------

def weighted_euclidean_distances(X: np.ndarray, query: np.ndarray, variable_weight: Optional[np.ndarray] = None, scale: bool = True) -> np.ndarray:
    X = _as_2d(X)
    query = np.asarray(query, dtype=float).reshape(1, -1)
    if query.shape[1] != X.shape[1]:
        raise ValueError("query must have the same number of columns as X")
    if scale:
        Xs, mu, std = standardize_fit(X)
        qs = (query - mu) / std
    else:
        Xs = X
        qs = query
    diff = Xs - qs
    if variable_weight is not None:
        w = np.asarray(variable_weight, dtype=float).reshape(1, -1)
        diff = diff * np.sqrt(np.clip(w, 0, np.inf))
    return np.sqrt(np.sum(diff**2, axis=1))


def nearest_neighbor_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    X_train = _as_2d(X_train)
    y_train = _as_1d(y_train)
    X_query = _as_2d(X_query)
    preds = []
    for q in X_query:
        d = weighted_euclidean_distances(X_train, q, scale=True)
        preds.append(y_train[int(np.argmin(d))])
    return np.asarray(preds)


def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int = 15, weighted: bool = True) -> np.ndarray:
    X_train = _as_2d(X_train)
    y_train = _as_1d(y_train)
    X_query = _as_2d(X_query)
    k = min(max(1, int(k)), X_train.shape[0])
    preds = []
    for q in X_query:
        d = weighted_euclidean_distances(X_train, q, scale=True)
        idx = np.argsort(d)[:k]
        if weighted:
            w = 1.0 / (d[idx] + EPS)
            preds.append(float(np.sum(w * y_train[idx]) / np.sum(w)))
        else:
            preds.append(float(np.mean(y_train[idx])))
    return np.asarray(preds)


def local_linear_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int = 35, alpha: float = 1e-3) -> np.ndarray:
    X_train = _as_2d(X_train)
    y_train = _as_1d(y_train)
    X_query = _as_2d(X_query)
    k = min(max(3, int(k)), X_train.shape[0])
    preds = []
    for q in X_query:
        d = weighted_euclidean_distances(X_train, q, scale=True)
        idx = np.argsort(d)[:k]
        h = np.median(d[idx]) + EPS
        w = np.exp(-0.5 * (d[idx] / h) ** 2)
        X_loc = X_train[idx]
        y_loc = y_train[idx]
        # Weighted local ridge in centered/scaled local coordinates.
        x_mean, x_std = _weighted_mean_std(X_loc, w)
        y_mean_arr, y_std_arr = _weighted_mean_std(y_loc.reshape(-1, 1), w)
        y_mean = float(y_mean_arr[0])
        y_std = float(y_std_arr[0])
        Xs = (X_loc - x_mean) / x_std
        ys = (y_loc - y_mean) / y_std
        sqrt_w = np.sqrt(w / (np.mean(w) + EPS))
        Xw = Xs * sqrt_w[:, None]
        yw = ys * sqrt_w
        p = Xw.shape[1]
        beta_std = np.linalg.solve(Xw.T @ Xw + alpha * np.eye(p), Xw.T @ yw)
        q_std = (q - x_mean) / x_std
        preds.append(float(y_mean + y_std * (q_std @ beta_std)))
    return np.asarray(preds)


def locally_weighted_pls_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, n_components: int = 3, bandwidth: float = 1.0, min_weight: float = 1e-6) -> np.ndarray:
    X_train = _as_2d(X_train)
    y_train = _as_1d(y_train)
    X_query = _as_2d(X_query)
    preds = []
    for q in X_query:
        d = weighted_euclidean_distances(X_train, q, scale=True)
        positive = d[d > EPS]
        h = bandwidth * (np.median(positive) if positive.size else 1.0)
        h = max(h, EPS)
        w = np.exp(-0.5 * (d / h) ** 2)
        w[w < min_weight] = 0.0
        if np.count_nonzero(w) < max(4, n_components + 2):
            idx = np.argsort(d)[: max(4, n_components + 2)]
            w = np.zeros_like(w)
            w[idx] = 1.0
        model = pls1_fit(X_train, y_train, n_components=n_components, sample_weight=w)
        preds.append(float(model.predict(q.reshape(1, -1))[0]))
    return np.asarray(preds)


# ---------------------------------------------------------------------------
# Gray-box modeling
# ---------------------------------------------------------------------------

def graybox_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[["feed", "temperature_c", "pressure", "residence_time", "impurity", "age_since_cleaning"]].to_numpy(float)
    # Add nonlinear terms that a black-box/residual model can exploit.
    temp = X[:, 1]
    pressure = X[:, 2]
    feed = X[:, 0]
    age = X[:, 5]
    return np.column_stack([X, np.sin(feed * 4.2), (pressure - 1.1) ** 2, temp - np.mean(temp), age / 100.0])


def fit_graybox_models(train_df: pd.DataFrame, y_train: Sequence[float], test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    y_train = _as_1d(y_train)
    g_train = train_df["physics_signal_g"].to_numpy(float)
    g_test = test_df["physics_signal_g"].to_numpy(float)
    # White-box: only a single scalar parameter with known physics signal.
    theta_hat = float(np.dot(g_train, y_train) / (np.dot(g_train, g_train) + EPS))
    pred_white = theta_hat * g_test

    X_train = graybox_feature_matrix(train_df)
    X_test = graybox_feature_matrix(test_df)
    black = ridge_fit(X_train, y_train, alpha=2.0, standardize=True)
    pred_black = black.predict(X_test)

    # Parallel: physics first, then model the remaining residual.
    pred_train_white = theta_hat * g_train
    residual_train = y_train - pred_train_white
    residual_model = ridge_fit(X_train, residual_train, alpha=1.0, standardize=True)
    pred_parallel = pred_white + residual_model.predict(X_test)

    # Serial: estimate local theta from data and learn how theta changes.
    theta_sample = y_train / (g_train + EPS)
    # Robustly clip extreme theta estimates caused by small g or noise.
    lo, hi = np.quantile(theta_sample, [0.03, 0.97])
    theta_sample = np.clip(theta_sample, lo, hi)
    theta_model = ridge_fit(X_train, theta_sample, alpha=3.0, standardize=True)
    pred_serial = theta_model.predict(X_test) * g_test

    # Combined: serial parameter adaptation plus residual correction.
    pred_train_serial = theta_model.predict(X_train) * g_train
    residual2 = y_train - pred_train_serial
    residual_model2 = ridge_fit(X_train, residual2, alpha=1.0, standardize=True)
    pred_combined = pred_serial + residual_model2.predict(X_test)

    return {
        "white_physics_only": pred_white,
        "black_statistical": pred_black,
        "parallel_graybox": pred_parallel,
        "serial_graybox": pred_serial,
        "combined_graybox": pred_combined,
    }


# ---------------------------------------------------------------------------
# Transfer learning / heterogeneous FEDA-style feature expansion
# ---------------------------------------------------------------------------

def fehda_expand_source(Cs: np.ndarray, Us: np.ndarray, n_target_unique: int) -> np.ndarray:
    Cs = _as_2d(Cs)
    Us = _as_2d(Us)
    n, n_common = Cs.shape
    return np.column_stack([
        Cs,
        Cs,
        Us,
        np.zeros((n, n_common)),
        np.zeros((n, n_target_unique)),
    ])


def fehda_expand_target(Ct: np.ndarray, Ut: np.ndarray, n_source_unique: int) -> np.ndarray:
    Ct = _as_2d(Ct)
    Ut = _as_2d(Ut)
    n, n_common = Ct.shape
    return np.column_stack([
        Ct,
        np.zeros((n, n_common)),
        np.zeros((n, n_source_unique)),
        Ct,
        Ut,
    ])


def bagged_ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 5.0, n_models: int = 80, sample_fraction: float = 0.85, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    X_train = _as_2d(X_train)
    y_train = _as_1d(y_train)
    X_test = _as_2d(X_test)
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    m = max(2, int(round(n * sample_fraction)))
    preds = []
    for _ in range(n_models):
        idx = rng.integers(0, n, size=m)
        mdl = ridge_fit(X_train[idx], y_train[idx], alpha=alpha, standardize=True)
        preds.append(mdl.predict(X_test))
    pred_mat = np.vstack(preds)
    return np.mean(pred_mat, axis=0), np.std(pred_mat, axis=0, ddof=1)


def transfer_learning_benchmark(dom: Dict[str, np.ndarray], alpha: float = 8.0, seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    Cs, Us, ys = dom["Cs"], dom["Us"], dom["ys"]
    Ct_train, Ut_train, yt_train = dom["Ct_train"], dom["Ut_train"], dom["yt_train"]
    Ct_test, Ut_test, yt_test = dom["Ct_test"], dom["Ut_test"], dom["yt_test"]
    n_us = Us.shape[1]
    n_ut = Ut_train.shape[1]

    predictions: Dict[str, np.ndarray] = {}

    # Target only: realistic but data-starved.
    X_t_train = np.column_stack([Ct_train, Ut_train])
    X_t_test = np.column_stack([Ct_test, Ut_test])
    mdl_target = ridge_fit(X_t_train, yt_train, alpha=alpha, standardize=True)
    predictions["target_only"] = mdl_target.predict(X_t_test)

    # Source-only common variables: ignores target-specific sensors and domain shift.
    mdl_source_common = ridge_fit(Cs, ys, alpha=alpha, standardize=True)
    predictions["source_only_common"] = mdl_source_common.predict(Ct_test)

    # Both-domain common variables: uses target data but discards unique variables.
    X_common = np.vstack([Cs, Ct_train])
    y_common = np.r_[ys, yt_train]
    mdl_both_common = ridge_fit(X_common, y_common, alpha=alpha, standardize=True)
    predictions["both_domain_common"] = mdl_both_common.predict(Ct_test)

    # Heterogeneous FEDA-style expansion.
    Xs_aug = fehda_expand_source(Cs, Us, n_target_unique=n_ut)
    Xt_aug = fehda_expand_target(Ct_train, Ut_train, n_source_unique=n_us)
    Xtest_aug = fehda_expand_target(Ct_test, Ut_test, n_source_unique=n_us)
    X_aug = np.vstack([Xs_aug, Xt_aug])
    y_aug = np.r_[ys, yt_train]
    # Weight the scarce target rows by duplicating them: simple and transparent.
    target_repeat = max(1, int(round(Cs.shape[0] / max(1, Ct_train.shape[0]) / 4)))
    if target_repeat > 1:
        X_aug = np.vstack([Xs_aug, np.repeat(Xt_aug, target_repeat, axis=0)])
        y_aug = np.r_[ys, np.repeat(yt_train, target_repeat)]
    pred_mean, pred_std = bagged_ridge_predict(X_aug, y_aug, Xtest_aug, alpha=alpha, n_models=100, sample_fraction=0.85, seed=seed)
    predictions["fehda_bagged_ridge"] = pred_mean
    predictions["fehda_uncertainty_std"] = pred_std

    table = metrics_table(yt_test, {k: v for k, v in predictions.items() if k != "fehda_uncertainty_std"})
    return table, predictions


# ---------------------------------------------------------------------------
# Raw data diagnostics
# ---------------------------------------------------------------------------

def robust_z_score(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < EPS:
        std = safe_std(x, axis=0, ddof=1)
        return (x - np.nanmean(x)) / (float(std) + EPS)
    return 0.6745 * (x - med) / mad


def count_histogram_peaks(x: np.ndarray, bins: int = 24) -> int:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0
    hist, _ = np.histogram(x, bins=bins)
    # Smooth by a tiny moving average.
    smooth = np.convolve(hist.astype(float), np.ones(3) / 3.0, mode="same")
    peaks = 0
    threshold = max(2.0, 0.15 * np.max(smooth))
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] >= threshold:
            peaks += 1
    return int(peaks)


def max_autocorrelation(x: np.ndarray, max_lag: int = 80) -> Tuple[int, float]:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    x = x[mask]
    if len(x) < 20:
        return 0, 0.0
    x = x - np.mean(x)
    denom = np.dot(x, x) + EPS
    max_lag = min(max_lag, len(x) // 3)
    best_lag, best_ac = 0, 0.0
    for lag in range(2, max_lag + 1):
        ac = float(np.dot(x[:-lag], x[lag:]) / denom)
        if abs(ac) > abs(best_ac):
            best_lag, best_ac = lag, ac
    return best_lag, best_ac


def diagnose_raw_data(df: pd.DataFrame, lower_clip_tolerance: float = 1e-9) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for col in df.columns:
        x = df[col].to_numpy(dtype=float)
        finite = np.isfinite(x)
        xf = x[finite]
        n = len(x)
        missing_rate = float(1.0 - np.mean(finite))
        if len(xf) < 5:
            rows.append({"variable": col, "missing_rate": missing_rate, "flags": "too_few_values"})
            continue
        z = robust_z_score(xf)
        outlier_rate = float(np.mean(np.abs(z) > 3.5))
        q25, q75 = np.quantile(xf, [0.25, 0.75])
        iqr = q75 - q25
        if iqr < EPS:
            iqr = float(np.std(xf) + EPS)
        diff = np.diff(xf)
        jump_score = float(np.max(np.abs(diff)) / (iqr + EPS)) if len(diff) else 0.0
        time_index = np.arange(n)[finite]
        trend_corr = float(np.corrcoef(time_index, xf)[0, 1]) if len(xf) > 3 and np.std(xf) > EPS else 0.0
        lag, ac = max_autocorrelation(xf)
        min_val = float(np.min(xf))
        lower_clip_rate = float(np.mean(np.abs(xf - min_val) <= lower_clip_tolerance + 1e-12))
        peaks = count_histogram_peaks(xf)
        flags = []
        if missing_rate > 0.03:
            flags.append("missing")
        if outlier_rate > 0.005:
            flags.append("outlier")
        if jump_score > 7.0:
            flags.append("jump_or_range_shift")
        if abs(trend_corr) > 0.65:
            flags.append("trend")
        if abs(ac) > 0.45 and lag > 2:
            flags.append("periodic_candidate")
        if lower_clip_rate > 0.05:
            flags.append("lower_or_upper_limit")
        if peaks >= 2:
            flags.append("multi_modal")
        rows.append(
            {
                "variable": col,
                "missing_rate": missing_rate,
                "outlier_rate": outlier_rate,
                "jump_score": jump_score,
                "trend_corr_with_time": trend_corr,
                "best_autocorr_lag": lag,
                "best_autocorr": ac,
                "limit_repeat_rate": lower_clip_rate,
                "histogram_peaks": peaks,
                "flags": ", ".join(flags) if flags else "ok",
            }
        )
    summary = pd.DataFrame(rows)

    corr = df.corr(numeric_only=True).abs()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if np.isfinite(val) and val >= 0.85:
                pairs.append({"variable_1": cols[i], "variable_2": cols[j], "abs_correlation": float(val)})
    high_corr = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False).reset_index(drop=True) if pairs else pd.DataFrame(columns=["variable_1", "variable_2", "abs_correlation"])
    return summary, high_corr


# ---------------------------------------------------------------------------
# End-to-end experiment wrappers
# ---------------------------------------------------------------------------

def experiment_regression_basics(seed: int = 1) -> Dict[str, object]:
    X, y = make_simple_regression(seed=seed, corr_strength=0.72)
    model = ols_fit(X, y, standardize=True)
    corr = float(np.corrcoef(X[:, 0], y)[0, 1])
    return {
        "X": X,
        "y": y,
        "standardized_ols_slope": float(model.coef_std[0]),
        "correlation": corr,
        "absolute_difference": abs(float(model.coef_std[0]) - corr),
    }


def experiment_multicollinearity(seed: int = 2) -> Dict[str, object]:
    X, y, names, true_coef = make_multicollinearity_data(seed=seed)
    X2 = X + np.random.default_rng(seed + 99).normal(scale=0.04, size=X.shape)
    ols1 = ols_fit(X, y, standardize=True)
    ols2 = ols_fit(X2, y, standardize=True)
    rows = []
    rows.append({"dataset": "A", "model": "OLS", **{names[i]: ols1.coef_std[i] for i in range(len(names))}})
    rows.append({"dataset": "B", "model": "OLS", **{names[i]: ols2.coef_std[i] for i in range(len(names))}})
    for r in [1, 2, 3]:
        pls_a = pls1_fit(X, y, n_components=r)
        pls_b = pls1_fit(X2, y, n_components=r)
        rows.append({"dataset": "A", "model": f"PLS R={r}", **{names[i]: pls_a.coef_std[i] for i in range(len(names))}})
        rows.append({"dataset": "B", "model": f"PLS R={r}", **{names[i]: pls_b.coef_std[i] for i in range(len(names))}})
    coef_table = pd.DataFrame(rows)
    cv = pls_cross_validate(X, y, max_components=3, k=5, seed=seed)
    vif = vif_table(X, names)
    return {"X": X, "X_noisy_copy": X2, "y": y, "names": names, "true_coef": true_coef, "coef_table": coef_table, "cv": cv, "vif": vif, "condition_number": condition_number(X)}


def experiment_jit_soft_sensor(seed: int = 3) -> Dict[str, object]:
    df, y = make_soft_sensor_process(seed=seed)
    X = df.to_numpy(float)
    y_arr = y.to_numpy(float)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y_arr, train_fraction=0.72)
    models = {
        "global_PLS": pls1_fit(X_train, y_train, n_components=4).predict(X_test),
        "nearest_neighbor": nearest_neighbor_predict(X_train, y_train, X_test),
        "kNN_weighted": knn_predict(X_train, y_train, X_test, k=18, weighted=True),
        "local_linear": local_linear_predict(X_train, y_train, X_test, k=80, alpha=0.1),
        "locally_weighted_PLS": locally_weighted_pls_predict(X_train, y_train, X_test, n_components=2, bandwidth=0.25),
    }
    table = metrics_table(y_test, models)
    return {"df": df, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "predictions": models, "metrics": table}


def experiment_graybox(seed: int = 4) -> Dict[str, object]:
    df, y, theta_true = make_graybox_process(seed=seed)
    n_train = int(len(y) * 0.7)
    train_df, test_df = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    y_train, y_test = y.iloc[:n_train].to_numpy(), y.iloc[n_train:].to_numpy()
    preds = fit_graybox_models(train_df, y_train, test_df)
    table = metrics_table(y_test, preds)
    return {"df": df, "y": y, "theta_true": theta_true, "train_df": train_df, "test_df": test_df, "y_train": y_train, "y_test": y_test, "predictions": preds, "metrics": table}


def experiment_transfer_learning(seed: int = 5) -> Dict[str, object]:
    dom = make_transfer_domains(seed=seed)
    table, preds = transfer_learning_benchmark(dom, seed=seed)
    return {"domains": dom, "metrics": table, "predictions": preds}


def experiment_raw_data(seed: int = 6) -> Dict[str, object]:
    df = make_raw_data_diagnostic_case(seed=seed)
    summary, high_corr = diagnose_raw_data(df)
    return {"df": df, "summary": summary, "high_correlation_pairs": high_corr}
