"""Run all process-data modeling simulations and export results.

Usage:
    python process_modeling_lab.py --out outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_modeling_core import (
    experiment_graybox,
    experiment_jit_soft_sensor,
    experiment_multicollinearity,
    experiment_raw_data,
    experiment_regression_basics,
    experiment_transfer_learning,
    ols_fit,
    pcr_cross_validate,
    pls_cross_validate,
    r2_score,
    rmse,
)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_regression_basics(res: dict, out: Path) -> None:
    X, y = res["X"], res["y"]
    model = ols_fit(X, y, standardize=True)
    xs = np.linspace(float(np.min(X)), float(np.max(X)), 100).reshape(-1, 1)
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], y, alpha=0.75, label="data")
    plt.plot(xs[:, 0], model.predict(xs), label="OLS")
    plt.title("Single regression: standardized slope equals correlation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "01_regression_basics.png", dpi=160)
    plt.close()


def plot_multicollinearity(res: dict, out: Path) -> None:
    coef_table = res["coef_table"].copy()
    rows = coef_table[coef_table["model"].isin(["OLS", "PLS R=1", "PLS R=2", "PLS R=3"])].copy()
    rows["label"] = rows["dataset"] + " / " + rows["model"]
    plot_df = rows.set_index("label")[["x1", "x2", "x3"]]
    ax = plot_df.plot(kind="bar", figsize=(10, 5))
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Coefficient stability under multicollinearity")
    ax.set_ylabel("standardized coefficient")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "02_multicollinearity_coefficients.png", dpi=160)
    plt.close()

    cv = res["cv"]
    plt.figure(figsize=(7, 5))
    plt.plot(cv["components"], cv["CV_RMSE"], marker="o", label="PLS")
    plt.title("PLS component selection by cross-validation")
    plt.xlabel("number of latent variables")
    plt.ylabel("CV RMSE")
    plt.xticks(cv["components"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "03_pls_cross_validation.png", dpi=160)
    plt.close()


def plot_jit(res: dict, out: Path) -> None:
    y_test = res["y_test"]
    preds = res["predictions"]
    plt.figure(figsize=(11, 5))
    plt.plot(y_test, label="actual", linewidth=2)
    for name in ["global_PLS", "kNN_weighted", "local_linear", "locally_weighted_PLS"]:
        plt.plot(preds[name], label=name, alpha=0.9)
    plt.title("Soft-sensor prediction on future samples")
    plt.xlabel("test sample index")
    plt.ylabel("quality")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out / "04_jit_soft_sensor_predictions.png", dpi=160)
    plt.close()


def plot_graybox(res: dict, out: Path) -> None:
    y_test = res["y_test"]
    preds = res["predictions"]
    plt.figure(figsize=(7, 6))
    lo = min(float(np.min(y_test)), *(float(np.min(v)) for v in preds.values()))
    hi = max(float(np.max(y_test)), *(float(np.max(v)) for v in preds.values()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="ideal")
    for name in ["white_physics_only", "black_statistical", "parallel_graybox", "combined_graybox"]:
        plt.scatter(y_test, preds[name], alpha=0.65, label=name)
    plt.title("Physical, statistical, and gray-box models")
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "05_graybox_actual_vs_predicted.png", dpi=160)
    plt.close()


def plot_transfer(res: dict, out: Path) -> None:
    dom = res["domains"]
    yt = dom["yt_test"]
    preds = res["predictions"]
    pred = preds["fehda_bagged_ridge"]
    std = preds["fehda_uncertainty_std"]
    idx = np.arange(len(yt))
    plt.figure(figsize=(11, 5))
    plt.plot(idx, yt, label="actual")
    plt.plot(idx, pred, label="FEHDA-style prediction")
    plt.fill_between(idx, pred - 1.96 * std, pred + 1.96 * std, alpha=0.2, label="approx. uncertainty band")
    plt.title("Transfer learning with scarce target data")
    plt.xlabel("target test sample")
    plt.ylabel("quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "06_transfer_learning_prediction.png", dpi=160)
    plt.close()


def plot_raw_data(res: dict, out: Path) -> None:
    df = res["df"]
    cols = ["outlier_like", "range_shift", "trend", "periodic", "missing_block", "lower_clipped"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 10), sharex=True)
    for ax, col in zip(axes, cols):
        ax.plot(df[col].to_numpy())
        ax.set_ylabel(col)
    axes[-1].set_xlabel("sample index")
    fig.suptitle("Raw data patterns to inspect before modeling")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out / "07_raw_data_patterns.png", dpi=160)
    plt.close(fig)


def run_all(out_dir: Path, seed: int = 1, make_plots: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    regression = experiment_regression_basics(seed=seed)
    save_table(
        pd.DataFrame([
            {
                "standardized_ols_slope": regression["standardized_ols_slope"],
                "correlation": regression["correlation"],
                "absolute_difference": regression["absolute_difference"],
            }
        ]),
        out_dir / "01_regression_basics_summary.csv",
    )

    multi = experiment_multicollinearity(seed=seed + 1)
    save_table(multi["coef_table"], out_dir / "02_multicollinearity_coefficients.csv")
    save_table(multi["cv"], out_dir / "03_pls_cross_validation.csv")
    save_table(multi["vif"], out_dir / "04_vif_table.csv")
    save_table(pd.DataFrame([{"condition_number": multi["condition_number"]}]), out_dir / "05_condition_number.csv")

    jit = experiment_jit_soft_sensor(seed=seed + 2)
    save_table(jit["metrics"], out_dir / "06_jit_soft_sensor_metrics.csv")

    gray = experiment_graybox(seed=seed + 3)
    save_table(gray["metrics"], out_dir / "07_graybox_metrics.csv")

    transfer = experiment_transfer_learning(seed=seed + 4)
    save_table(transfer["metrics"], out_dir / "08_transfer_learning_metrics.csv")

    raw = experiment_raw_data(seed=seed + 5)
    save_table(raw["summary"], out_dir / "09_raw_data_diagnostics.csv")
    save_table(raw["high_correlation_pairs"], out_dir / "10_high_correlation_pairs.csv")

    if make_plots:
        plot_regression_basics(regression, out_dir)
        plot_multicollinearity(multi, out_dir)
        plot_jit(jit, out_dir)
        plot_graybox(gray, out_dir)
        plot_transfer(transfer, out_dir)
        plot_raw_data(raw, out_dir)

    print(f"Saved results to: {out_dir}")
    print("\nJIT soft-sensor metrics")
    print(jit["metrics"].to_string(index=False))
    print("\nGray-box metrics")
    print(gray["metrics"].to_string(index=False))
    print("\nTransfer-learning metrics")
    print(transfer["metrics"].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    args = parser.parse_args()
    run_all(args.out, seed=args.seed, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
