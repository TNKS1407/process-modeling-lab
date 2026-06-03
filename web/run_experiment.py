"""Run one process-modeling experiment and save PNGs to output dir.

Usage:
    python run_experiment.py <exp_name> <output_dir>
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from process_modeling_core import (
    experiment_graybox,
    experiment_jit_soft_sensor,
    experiment_multicollinearity,
    experiment_raw_data,
    experiment_regression_basics,
    experiment_transfer_learning,
    ols_fit,
)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run_regression(out: Path) -> list[str]:
    res = experiment_regression_basics()
    X, y = res["X"], res["y"]
    corr = res["correlation"]
    slope = res["standardized_ols_slope"]
    xs = np.linspace(float(np.min(X)), float(np.max(X)), 100).reshape(-1, 1)
    model = ols_fit(X, y, standardize=True)
    plt.figure(figsize=(7, 4))
    plt.scatter(X[:, 0], y, alpha=0.7, s=20, label="data")
    plt.plot(xs[:, 0], model.predict(xs), label=f"OLS slope={slope:.4f}")
    plt.title(f"単回帰: 標準化傾き={slope:.4f}  相関係数={corr:.4f}  差={res['absolute_difference']:.6f}")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    save_fig(out / "01_regression.png")
    return ["01_regression.png"]


def run_multicollinearity(out: Path) -> list[str]:
    res = experiment_multicollinearity()
    coef_table = res["coef_table"]
    rows = coef_table[coef_table["model"].isin(["OLS", "PLS R=1", "PLS R=2", "PLS R=3"])].copy()
    rows["label"] = rows["dataset"] + " / " + rows["model"]
    feat_cols = [c for c in ["x1","x2","x3"] if c in rows.columns]
    if feat_cols:
        plot_df = rows.set_index("label")[feat_cols]
        ax = plot_df.plot(kind="bar", figsize=(10, 5))
        ax.axhline(0, linewidth=0.8)
        ax.set_title("多重共線性下での係数安定性（OLS vs PLS）")
        ax.set_ylabel("標準化係数")
        plt.xticks(rotation=45, ha="right")
        save_fig(out / "02_multicollinearity.png")

    cv = res["cv"]
    plt.figure(figsize=(7, 4))
    plt.plot(cv["components"], cv["CV_RMSE"], marker="o")
    plt.title("PLS 潜在変数数の交差検証")
    plt.xlabel("潜在変数数"); plt.ylabel("CV RMSE")
    save_fig(out / "03_pls_cv.png")
    return ["02_multicollinearity.png", "03_pls_cv.png"]


def run_jit(out: Path) -> list[str]:
    res = experiment_jit_soft_sensor()
    y_test = np.array(res["y_test"])
    plt.figure(figsize=(10, 5))
    for name, preds in res["predictions"].items():
        err = np.array(preds) - y_test
        plt.plot(err, label=name, alpha=0.8)
    plt.axhline(0, linewidth=0.8, color="k")
    plt.title("JIT型ソフトセンサー: 予測誤差比較")
    plt.xlabel("テストサンプル"); plt.ylabel("残差"); plt.legend(fontsize=8)
    save_fig(out / "04_jit_residuals.png")

    metrics = res["metrics"]
    fig, ax = plt.subplots(figsize=(7, max(2, len(metrics)*0.45 + 1)))
    ax.axis("off")
    if isinstance(metrics, pd.DataFrame):
        data = metrics.values.tolist()
        cols = metrics.columns.tolist()
    else:
        data = [[m, f"{v['RMSE']:.4f}", f"{v['MAE']:.4f}", f"{v['R2']:.4f}"] for m, v in metrics.items()]
        cols = ["モデル", "RMSE", "MAE", "R²"]
    tbl = ax.table(cellText=[[str(c) for c in r] for r in data], colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(True)
    plt.title("モデル比較")
    save_fig(out / "04_jit_metrics.png")
    return ["04_jit_residuals.png", "04_jit_metrics.png"]


def run_graybox(out: Path) -> list[str]:
    res = experiment_graybox()
    y_test = np.array(res["y_test"])
    plt.figure(figsize=(10, 5))
    for name, preds in res["predictions"].items():
        plt.plot(np.array(preds), label=name, alpha=0.8)
    plt.plot(y_test, "k--", label="実測", linewidth=1.5)
    plt.title("グレーボックスモデル: 予測vs実測")
    plt.xlabel("サンプル"); plt.ylabel("値"); plt.legend(fontsize=8)
    save_fig(out / "05_graybox_preds.png")

    metrics = res["metrics"]
    fig, ax = plt.subplots(figsize=(7, max(2, len(metrics)*0.45 + 1)))
    ax.axis("off")
    if isinstance(metrics, pd.DataFrame):
        data = metrics.values.tolist()
        cols = metrics.columns.tolist()
    else:
        data = [[m, f"{v['RMSE']:.4f}", f"{v['MAE']:.4f}", f"{v['R2']:.4f}"] for m, v in metrics.items()]
        cols = ["モデル", "RMSE", "MAE", "R²"]
    tbl = ax.table(cellText=[[str(c) for c in r] for r in data], colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(True)
    plt.title("グレーボックス比較")
    save_fig(out / "05_graybox_metrics.png")
    return ["05_graybox_preds.png", "05_graybox_metrics.png"]


def run_transfer(out: Path) -> list[str]:
    res = experiment_transfer_learning()
    domains = res["domains"]
    y_test = np.array(domains["yt_test"])
    plt.figure(figsize=(10, 5))
    for name, preds in res["predictions"].items():
        if name == "fehda_uncertainty_std":
            continue
        plt.plot(np.array(preds), label=name, alpha=0.8)
    plt.plot(y_test, "k--", label="実測", linewidth=1.5)
    plt.title("転移学習: 予測vs実測")
    plt.xlabel("サンプル"); plt.ylabel("値"); plt.legend(fontsize=8)
    save_fig(out / "06_transfer_preds.png")

    metrics = res["metrics"]
    fig, ax = plt.subplots(figsize=(7, max(2, len(metrics)*0.45 + 1)))
    ax.axis("off")
    if isinstance(metrics, pd.DataFrame):
        data = [[str(c) for c in r] for r in metrics.values.tolist()]
        cols = metrics.columns.tolist()
    else:
        data = [[m, f"{v['RMSE']:.4f}", f"{v['MAE']:.4f}", f"{v['R2']:.4f}"] for m, v in metrics.items()]
        cols = ["モデル", "RMSE", "MAE", "R²"]
    tbl = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(True)
    plt.title("転移学習比較")
    save_fig(out / "06_transfer_metrics.png")
    return ["06_transfer_preds.png", "06_transfer_metrics.png"]


def run_rawdata(out: Path) -> list[str]:
    res = experiment_raw_data()
    df = res["df"]
    summary = res["summary"]

    # Plot time series for each variable
    cols = df.columns.tolist()
    ncols = min(len(cols), 8)
    nrows = (ncols + 3) // 4
    fig, axes = plt.subplots(nrows, 4, figsize=(14, nrows * 3 + 1))
    axes_flat = np.array(axes).flat
    for i, col in enumerate(cols[:ncols]):
        ax = next(axes_flat)
        ax.plot(df[col].values, linewidth=0.7)
        if summary is not None and col in summary["variable"].values:
            row = summary[summary["variable"] == col].iloc[0]
            flags = row.get("flags", "")
            color = "#e05050" if flags else "#27ae60"
            label = f"⚠ {flags}" if flags else "✓ OK"
        else:
            color, label = "#7a7f9a", ""
        ax.set_title(f"{col}\n{label}", fontsize=8, color=color)
        ax.tick_params(labelsize=7)
    for ax in axes_flat:
        ax.set_visible(False)
    plt.suptitle("生データ診断: 時系列", fontsize=12)
    save_fig(out / "07_rawdata_series.png")

    # Summary table
    if summary is not None and not summary.empty:
        show_cols = ["variable", "missing_rate", "outlier_rate", "flags"]
        show_cols = [c for c in show_cols if c in summary.columns]
        fig, ax = plt.subplots(figsize=(10, max(2, len(summary)*0.4 + 1)))
        ax.axis("off")
        disp = summary[show_cols].copy()
        for c in ["missing_rate", "outlier_rate"]:
            if c in disp.columns:
                disp[c] = disp[c].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
        tbl = ax.table(cellText=disp.values.tolist(), colLabels=disp.columns.tolist(),
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(True)
        plt.title("診断サマリー")
        save_fig(out / "07_rawdata_summary.png")
        return ["07_rawdata_series.png", "07_rawdata_summary.png"]

    return ["07_rawdata_series.png"]


EXPERIMENTS = {
    "regression": run_regression,
    "multicollinearity": run_multicollinearity,
    "jit": run_jit,
    "graybox": run_graybox,
    "transfer": run_transfer,
    "rawdata": run_rawdata,
}

if __name__ == "__main__":
    exp_name = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = EXPERIMENTS.get(exp_name)
    if fn is None:
        print(f"Unknown experiment: {exp_name}", file=sys.stderr)
        sys.exit(1)
    files = fn(out_dir)
    print("\n".join(files))
