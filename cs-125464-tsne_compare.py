"""
3D t-SNE comparison across input sources for fog-visibility classification.

Purpose:
    - Load features/labels for SC, EC_low, and EC_high sources.
    - Optionally balance classes before projection.
    - Run 3D t-SNE per source and render side-by-side panels.

Inputs:
    - File paths and visualization parameters defined in this script.

Outputs:
    - A saved figure showing class separability across sources.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import resample
from mpl_toolkits.mplot3d import Axes3D  

SC_PATH = r"D:\hainan\regression\sc\sample\5_stations\samples1.npy"     
EC_LOW_PATH = r"D:\hainan\regression\ec_low_all\samples1.npy"             
EC_HIGH_PATH = r"D:\hainan\features\upper_air\feature1.npy"              
LABEL_PATH = r"D:\fog_prediction\data\ec_high\ViT_label\main_station\low\label\label1.npy"

OUT_DIR       = r"D:\hainan\figures"
FIG_NAME      = "tsne3d_compare_sc_eclow_echigh.png"

RANDOM_SEED   = 501
PERPLEXITY    = 30
LEARNING_RATE = 200
MAX_ITER      = 2000
USE_PCA_BEFORE_TSNE = True
PCA_N_COMPONENTS    = 50

BALANCE_CLASSES = True                     
COLOR_MAP = {0: "green", 1: "red"}         
POINT_SIZE = 30
ALPHA      = 0.7

ECHIGH_KEEP_N = 19120


def load_array(path):
    """
    Load a numpy array from disk and flatten to 2D if required by the pipeline.

    Inputs:
        - Path to a .npy file.

    Outputs:
        - A 2D array suitable for preprocessing and t-SNE.
    """
    arr = np.load(path, allow_pickle=True)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def balanced_subset(X, y, random_state=42):
    """
    Create a class-balanced subset by downsampling the majority class.

    Inputs:
        - Feature matrix and binary labels.
        - Optional random seed for reproducibility.

    Outputs:
        - Features and labels with balanced class counts.
    """
    y = y.astype(int).reshape(-1)
    cls0 = X[y == 0]
    cls1 = X[y == 1]
    if len(cls0) == 0 or len(cls1) == 0:
        return X, y
    n = min(len(cls0), len(cls1))
    cls0_ds = resample(cls0, replace=False, n_samples=n, random_state=random_state)
    cls1_ds = resample(cls1, replace=False, n_samples=n, random_state=random_state)
    Xb = np.vstack([cls0_ds, cls1_ds])
    yb = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    return Xb, yb


def tsne3d(X, y, ax, title,seed=RANDOM_SEED, perplexity=PERPLEXITY, lr=LEARNING_RATE, max_iter=MAX_ITER,use_pca=USE_PCA_BEFORE_TSNE, pca_n=PCA_N_COMPONENTS):
    """
    Project features via 3D t-SNE and render a class-colored scatter on the given axes.

    Inputs:
        - Features and binary labels.
        - A Matplotlib 3D axes object and a panel title.
        - t-SNE parameters (seed, perplexity, learning rate, iterations).
        - Optional PCA step before t-SSE.

    Outputs:
        - A rendered 3D scatter plot on the provided axes (returns None).
    """
    X_std = StandardScaler().fit_transform(X)

    X_in = X_std
    if use_pca and X_std.shape[1] > 3:
        n_comp = min(max(3, pca_n), max(3, X_std.shape[1] - 1))
        X_in = PCA(n_components=n_comp, random_state=seed).fit_transform(X_std)

    max_perf = max(5, len(X_in) // 5)
    perf = max(5, min(perplexity, max_perf))

    tsne = TSNE(
        n_components=3, random_state=seed,
        perplexity=perf, learning_rate=lr, max_iter=max_iter,
        init="pca", n_iter_without_progress=400, verbose=0
    )
    X3 = tsne.fit_transform(X_in) 

    for cls in (0, 1):
        m = (y == cls)
        if m.sum() == 0:
            continue
        ax.scatter(
            X3[m, 0], X3[m, 1], X3[m, 2],
            color=COLOR_MAP[cls],
            s=POINT_SIZE,
            alpha=ALPHA,
            linewidths=0,
            label=('Non-fog (≥1000 m)' if cls == 0 else 'Fog (<1000 m)')
        )

    ax.grid(True)
    ax.set_facecolor("white")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis._axinfo['grid'].update({
                'color': (0.7, 0.7, 0.7, 1),
                'linewidth': 0.8,
                'linestyle': '-'
            })
        except Exception:
            pass

    ax.locator_params(nbins=8, axis='x')
    ax.locator_params(nbins=8, axis='y')
    ax.locator_params(nbins=8, axis='z')

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_visible(False)
        except Exception:
            pass

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.tick_params(axis='x', which='both', length=0, width=0)
    ax.tick_params(axis='y', which='both', length=0, width=0)
    ax.tick_params(axis='z', which='both', length=0, width=0)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis._axinfo['tick']['inward_factor']  = 0.0
            axis._axinfo['tick']['linewidth']      = (0.0, 0.0)  
        except Exception:
            pass

    try:
        ax.xaxis.line.set_linewidth(1.6); ax.xaxis.line.set_color("black")
        ax.yaxis.line.set_linewidth(1.6); ax.yaxis.line.set_color("black")
        ax.zaxis.line.set_linewidth(1.6); ax.zaxis.line.set_color("black")
    except Exception:
        pass

    ax.view_init(elev=18, azim=20)

    ax.set_title(title, fontsize=12, pad=6)
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def main():
    """
    Load SC, EC_low, and EC_high features/labels, align sample counts,
    optionally balance classes, render three 3D t-SNE panels, and save the figure.

    Inputs:
        - Paths and parameters defined at the top of this script.

    Outputs:
        - A saved PNG figure and basic console logs.
    """

    os.makedirs(OUT_DIR, exist_ok=True)

    sc      = load_array(SC_PATH)         
    eclow   = load_array(EC_LOW_PATH)     
    echigh  = load_array(EC_HIGH_PATH)    
    labels  = np.load(LABEL_PATH, allow_pickle=True).astype(int).reshape(-1)

    keep_n = min(ECHIGH_KEEP_N, echigh.shape[0], labels.shape[0])
    echigh = echigh[:keep_n]
    labels = labels[:keep_n]

    sc    = sc[:labels.shape[0]]
    eclow = eclow[:labels.shape[0]]

    N = len(labels)
    assert sc.shape[0] == N and eclow.shape[0] == N and echigh.shape[0] == N, \
        f"Mismatch in sample counts: labels={N}, sc={sc.shape[0]}, eclow={eclow.shape[0]}, echigh={echigh.shape[0]}"
    print(f"[INFO] N={N}, d_sc={sc.shape[1]}, d_eclow={eclow.shape[1]}, d_echigh={echigh.shape[1]}")

    if BALANCE_CLASSES:
        sc_b,     y_sc  = balanced_subset(sc, labels)
        eclow_b,  y_low = balanced_subset(eclow, labels)
        echigh_b, y_hi  = balanced_subset(echigh, labels)
    else:
        sc_b, eclow_b, echigh_b = sc, eclow, echigh
        y_sc = y_low = y_hi = labels


    fig = plt.figure(figsize=(18, 6), constrained_layout=True)

    ax1 = fig.add_subplot(131, projection="3d")
    tsne3d(sc_b,     y_sc,  ax1, "SC Features (3D t-SNE)")

    ax2 = fig.add_subplot(132, projection="3d")
    tsne3d(eclow_b,  y_low, ax2, "EC_low Features (3D t-SNE)")

    ax3 = fig.add_subplot(133, projection="3d")
    tsne3d(echigh_b, y_hi,  ax3, "EC High-alt Features (3D t-SNE)")

    out_png = Path(OUT_DIR) / FIG_NAME
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[SAVE] 3D t-SNE comparison saved to: {out_png}")


if __name__ == "__main__":
    main()
