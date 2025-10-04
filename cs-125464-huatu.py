"""
Figure utilities for the manuscript.

Purpose:
    Generate accuracy-versus-horizon plots and pooled ROC/PR curves for multiple models.

Inputs:
    - Paths to metric files or directories containing saved predictions.
    - Display names and output directories.

Outputs:
    - Vector and raster figures suitable for publication.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scipy.interpolate import make_interp_spline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _smooth_xy(x, y, n_points=300):
    """
    Densify and smooth a polyline for presentation-quality figures.

    Inputs:
        - Original x and y sequences.
        - Desired number of interpolated points.

    Outputs:
        - Smoothed x and y sequences for plotting.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    mask = np.concatenate([[True], np.diff(x) > 0])
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return x, y
    x_new = np.linspace(x.min(), x.max(), n_points)
    if _HAS_SCIPY and len(x) >= 4:
        y_new = make_interp_spline(x, y, k=3)(x_new)
    else:
        y_new = np.interp(x_new, x, y)
    return x_new, y_new


def plot_accuracy(excel_path=r"C:\Users\Administrator\Desktop\Accuracy.xlsx",sheet_name=0,out_dir="./figures"):
    """
    Plot classification accuracy versus forecast horizon for multiple models.

    Inputs:
        - Path to a metrics workbook and the target sheet name or index.
        - Output directory for figures.

    Outputs:
        - Accuracy curves saved as PDF and PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
    df = df.apply(pd.to_numeric, errors="coerce")

    if np.nanmax(df.values) <= 1.2:
        df = df * 100.0

    x = np.linspace(0, 63, len(df))

    colors = [
        "#4C72B0",  
        "#DD8452",  
        "#55A868",  
        "#8172B3",  
        "#64B5CD",  
    ]

    order = ["T-LGB", "Adaboost", "Gbrt", "RandomForest", "XGBoost"]
    plt.figure(figsize=(9, 6))
    markers = ["^", "s", "D", "v", "o"]  

    plotted = []
    for name, color, mk in zip(order, colors, markers):
        if name not in df.columns:
            continue
        y = df[name].to_numpy()
        if np.all(np.isnan(y)):
            continue

        valid = ~np.isnan(y)
        xs, ys = _smooth_xy(x[valid], y[valid], n_points=300)
        plt.plot(xs, ys, linewidth=1.8, color=color)

        plt.scatter(
            x[valid], y[valid],
            marker=mk, s=22,
            color=color, zorder=3
        )
        plotted.append(name)


    ax = plt.gca()
    ax.margins(x=0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel("Horizon (hours)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Classification Accuracy — Model Comparison", fontsize=14)
    plt.xlim(-2, 63)
    plt.xticks(np.arange(0, 64, 12))


    legend_handles, legend_labels = [], []
    for name, color, mk in zip(order, colors, markers):
        if name not in plotted:
            continue
        handle = Line2D(
            [0], [0],
            color=color, linewidth=1.8,
            marker=mk, markersize=6,
            markerfacecolor=color, markeredgecolor=color
        )
        legend_handles.append(handle)
        legend_labels.append(name)
    fig = plt.gcf()

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_labels), frameon=False,
        handlelength=1.8, columnspacing=1.0, handletextpad=0.5
        )


    plt.tight_layout(rect=[0, 0, 1, 0.97])


    plt.savefig(out_dir / "accuracy_models.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "accuracy_models.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Saved to:", (out_dir / "accuracy_models.pdf").resolve())
    print("Saved to:", (out_dir / "accuracy_models.png").resolve())



from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def _load_pooled_truth_probs(model_dir):
    """
    Load and concatenate per-iteration ground truth and predicted probabilities.

    Inputs:
        - Directory containing saved arrays for ground truth and probabilities.

    Outputs:
        - Combined ground-truth sequence and combined probability sequence.
    """
    y_trues, y_probs = [], []
    for j in range(1, 22):
        yt_p = Path(model_dir) / f"y_true_iter{j}.npy"
        yp_p = Path(model_dir) / f"y_prob_iter{j}.npy"
        if yt_p.exists() and yp_p.exists():
            yt = np.load(yt_p)
            yp = np.load(yp_p)
            yt = yt.astype(int).reshape(-1)
            yp = yp.astype(float).reshape(-1)
            mask = np.isfinite(yp)
            y_trues.append(yt[mask])
            y_probs.append(yp[mask])
        else:
            print(f"[warning] Missing {yt_p.name} or {yp_p.name}; skipping.")
    if not y_trues:
        return None, None
    y_true_all = np.concatenate(y_trues, axis=0)
    y_prob_all = np.concatenate(y_probs, axis=0)
    return y_true_all, y_prob_all


def plot_roc_pr_for_models(model_dirs: dict,out_dir="./figures"):
    """
    Plot pooled ROC and Precision–Recall curves for a set of models.

    Inputs:
        - A mapping from display name to directory containing saved predictions.
        - Output directory for figures.

    Outputs:
        - ROC and PR figures with AUC/AP reported in the legend.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order   = ["T-LGB", "Adaboost", "Gbrt", "RandomForest", "XGBoost"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#64B5CD"]
    markers = ["^", "s", "D", "v", "o"]

    # ===== ROC =====
    plt.figure(figsize=(9, 6))
    legend_handles, legend_labels = [], []
    plotted = []

    for name, color, mk in zip(order, colors, markers):
        if name not in model_dirs:
            continue
        y_true, y_prob = _load_pooled_truth_probs(model_dirs[name])
        if y_true is None:
            print(f"[skip] {name} has no available y_true/y_prob files")
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, linewidth=1.8)
        idx = np.argmin(np.abs(tpr - 0.8))
        plt.scatter(fpr[idx], tpr[idx], marker=mk, s=28, color=color, zorder=3)
        plotted.append((name, color, mk, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC — Model Comparison", fontsize=14)

    from matplotlib.lines import Line2D
    for name, color, mk, roc_auc in plotted:
        handle = Line2D([0],[0], color=color, linewidth=1.8,
                        marker=mk, markersize=6,
                        markerfacecolor=color, markeredgecolor=color)
        legend_handles.append(handle)
        legend_labels.append(f"{name} (AUC={roc_auc:.3f})")
    fig = plt.gcf()
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_labels), frameon=False,
        handlelength=1.8, columnspacing=1.0, handletextpad=0.5
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_dir / "roc_models.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "roc_models.png", dpi=600, bbox_inches="tight")
    print("Saved:", (out_dir / "roc_models.png").resolve())

    # ===== PR =====
    plt.figure(figsize=(9, 6))
    legend_handles, legend_labels = [], []
    plotted = []

    for name, color, mk in zip(order, colors, markers):
        if name not in model_dirs:
            continue
        y_true, y_prob = _load_pooled_truth_probs(model_dirs[name])
        if y_true is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, color=color, linewidth=1.8)
        idx = np.argmin(np.abs(recall - 0.8))
        plt.scatter(recall[idx], precision[idx], marker=mk, s=28, color=color, zorder=3)
        plotted.append((name, color, mk, ap))

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision–Recall — Model Comparison", fontsize=14)

    from matplotlib.lines import Line2D
    for name, color, mk, ap in plotted:
        handle = Line2D([0],[0], color=color, linewidth=1.8,
                        marker=mk, markersize=6,
                        markerfacecolor=color, markeredgecolor=color)
        legend_handles.append(handle)
        legend_labels.append(f"{name} (AP={ap:.3f})")
    fig = plt.gcf()
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_labels), frameon=False,
        handlelength=1.8, columnspacing=1.0, handletextpad=0.5
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_dir / "pr_models.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "pr_models.png", dpi=600, bbox_inches="tight")
    print("Saved:", (out_dir / "pr_models.png").resolve())


if __name__ == "__main__":
    # plot_accuracy()
    model_dirs = {
        "T-LGB":        r"D:\hainan\results\T-LGB",
        "Adaboost":     r"D:\hainan\results\Adaboost",
        "Gbrt":         r"D:\hainan\results\GBDT",          
        "RandomForest": r"D:\hainan\results\RandomForest",
        "XGBoost":      r"D:\hainan\results\XGBoost",
    }
    plot_roc_pr_for_models(model_dirs, out_dir="./figures")