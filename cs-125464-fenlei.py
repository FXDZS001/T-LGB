"""
Classification metrics and analysis helpers for fog visibility.

Purpose:
    Provide correlation plotting, contingency-based metrics (TS, FAR, MAR),
    LightGBM feature-importance visualization, and utilities for saving
    predictions and aggregated metrics.

Inputs:
    - Observed labels and predicted probabilities or classes.
    - Paths for reading/writing figures and metric files.

Outputs:
    - Plots (e.g., correlation heatmaps, importance bars).
    - Scalar metrics and CSV summaries saved to disk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

import xgboost
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
warnings.filterwarnings("ignore") 


def plot_visibility_corr_heatmap(df: pd.DataFrame,
                                 out_png: str = "corr_heatmap.png",
                                 out_pdf: str = "corr_heatmap.pdf",
                                 title: str | None = None,
                                 var_order: list[str] | None = None,
                                 vis_col: str | None = None) -> None:
    """
    Plot a Pearson correlation heatmap with a diverging colormap (blue=positive, red=negative),
    centered at zero, value range [-1, 1], and annotated coefficients.

    Inputs:
        - df: a DataFrame containing visibility and meteorological variables.
        - out_png: path to save a high-DPI PNG (for submission portals).
        - out_pdf: path to save a vector PDF (for publication-quality).
        - title: optional figure title.
        - var_order: optional column order to display (use to align with the paper).
        - vis_col: optional column name for visibility (if you want to ensure its position).

    Outputs:
        - Two files saved to disk: PNG (600 dpi) and PDF (vector).
    """
    data = df.copy()
    if var_order is not None:
        cols = [c for c in var_order if c in data.columns]
        cols += [c for c in data.columns if c not in cols]
        data = data[cols]

    corr = data.corr(method="pearson")

    plt.figure(figsize=(8.5, 7.0))  
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",    
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar=True
    )

    if title:
        ax.set_title(title, pad=12)


    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)


    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.savefig(out_pdf)  
    print(f"[SAVE] Correlation heatmap saved to: {out_png} and {out_pdf}")

def prep_clf(obs,pre, threshold=0.1):
    """
    Convert probabilities to binary classes and compute contingency counts.

    Inputs:
        - Observed binary labels.
        - Predicted probabilities.
        - Decision threshold.

    Outputs:
        - Hit, miss, false-alarm, and correct-negative counts.
    """

    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives

def TS(obs, pre, threshold=0.1):
    """
    Compute the Threat Score (TS) from contingency counts.

    Inputs:
        - Observed labels, predicted probabilities or classes, and threshold.

    Outputs:
        - TS as a floating-point value.
    """
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return hits/(hits + falsealarms + misses)

def FAR(obs, pre, threshold=0.1):
    """
    Compute the False Alarm Rate (FAR) from contingency counts.

    Inputs:
        - Observed labels, predicted probabilities or classes, and threshold.

    Outputs:
        - FAR as a floating-point value.
    """
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)

def MAR(obs, pre, threshold=0.1):
    """
    Compute the Missing Alarm Rate (MAR) from contingency counts.

    Inputs:
        - Observed labels, predicted probabilities or classes, and threshold.

    Outputs:
        - MAR as a floating-point value.
    """
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return misses / (hits + misses)

def importance_feature():
    """
    Train a LightGBM model and visualize feature importances.

    Inputs:
        - Training features and labels.
        - Model parameters and display settings.

    Outputs:
        - A bar chart ranking features by gain, saved to disk.
    """
    root_path = r'F:\毕业'
    stations = ['总站']
    for i in range(len(stations)):
        for j in range(2, 3):
            # read_ec = os.path.join(root_path, stations[i], '分类', '低', 'high_ec', 'feature' + str(j) + '.npy')
            read_sc = os.path.join(root_path, 'Balanced', 'sc', 'sc' + str(j) + '.npy')
            read_label = os.path.join(root_path, 'Balanced', 'label', 'label' + str(j) + '.npy')
            read_eclow = os.path.join(root_path, 'Balanced', 'ec_low', 'ec_low' + str(j) + '.npy')
            sc = np.load(read_sc, allow_pickle=True)
            ec_low = np.load(read_eclow, allow_pickle=True)
            label = np.load(read_label, allow_pickle=True)
            label = label.squeeze()
            ec_low = ec_low.squeeze()
            sample = np.concatenate((sc, ec_low), axis=1)
            # sample = ec
            train_featrues = sample
            train_labels = label
            model = LGBMClassifier(objective='regression',
                                   max_depth=8,
                                   learning_rate=0.1, n_estimators=7850,
                                   bagging_fraction=0.85, feature_fraction=0.85)
            model.fit(train_featrues, train_labels.astype('int'))  

            plt.figure(figsize=(12, 6))
            lgb.plot_importance(model, xlabel='F score', max_num_features=30)
            plt.title("Featurertances")
            plt.show()

def get_train_x_y(sample,label):
    """
    Standardize features and return (X, y) for model training.

    Inputs:
        - sample: raw feature matrix.
        - label:  label vector.

    Outputs:
        - x: standardized features.
        - y: labels unchanged.
    """
    x = sample
    y = label
    x = StandardScaler().fit_transform(x)
    return x, y

import os, numpy as np, pandas as pd

def save_metrics_and_probs(model_name, save_root, iteration, y_test, y_pred, y_prob, metrics_dict):
    """
    Save per-iteration ground truth and predicted probabilities, and append
    scalar metrics into an in-memory dictionary.

    Inputs:
        - model_name: display name used for directory naming.
        - save_root:  root folder for outputs.
        - iteration:  current iteration index.
        - y_test, y_pred, y_prob: arrays of ground truth, predicted class, and probability.
        - metrics_dict: dict of lists to collect scalar metrics across iterations.

    Outputs:
        - Files y_true_iter{iteration}.npy and y_prob_iter{iteration}.npy saved to disk.
        - The same metrics_dict updated in-place; returns the model output directory.
    """

    model_dir = os.path.join(save_root, model_name)
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(model_dir, f"y_true_iter{iteration}.npy"), y_test.astype(int))
    np.save(os.path.join(model_dir, f"y_prob_iter{iteration}.npy"), y_prob.astype(float))


    metrics_dict['Iteration'].append(iteration)
    metrics_dict['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
    metrics_dict['Recall'].append(recall_score(y_test, y_pred) * 100)
    metrics_dict['F1_Score'].append(f1_score(y_test, y_pred) * 100)
    metrics_dict['TS_Score'].append(TS(y_test, y_pred) * 100)
    metrics_dict['FAR'].append(FAR(y_test, y_pred) * 100)
    metrics_dict['MAR'].append(MAR(y_test, y_pred) * 100)

    return model_dir  


def finalize_metrics(model_name, model_dir, metrics_dict):
    """
    Save aggregated metrics collected across iterations as a CSV file.

    Inputs:
        - model_name: display name for the model.
        - model_dir:  directory containing per-iteration outputs.
        - metrics_dict: dictionary with lists of scalar metrics.

    Outputs:
        - A CSV file with aggregated metrics written to the model directory.
    """
    df = pd.DataFrame(metrics_dict)
    out_csv = os.path.join(model_dir, f"metrics_{model_name}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {model_name} metrics CSV saved: {out_csv}")

def func_C():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = LGBMClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("T-LGB", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("T-LGB", model_dir, metrics)


def GBDT():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = GradientBoostingClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("GBDT", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("GBDT", model_dir, metrics)


def XGBoost():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = XGBClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("XGBoost", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("XGBoost", model_dir, metrics)


def RandomForest():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = RandomForestClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("RandomForest", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("RandomForest", model_dir, metrics)


def AdaBoost():
    root_path = r'D:\hainan'
    save_root = r"D:\hainan\results"
    metrics = {'Iteration': [], 'Accuracy': [], 'Recall': [], 
               'F1_Score': [], 'TS_Score': [], 'FAR': [], 'MAR': []}

    for j in range(1, 22):
        read_sample = os.path.join(root_path, 'results', 'merged_npy', f'sample{j}.npy')
        read_label = os.path.join(root_path, 'Balanced', 'label', f'label{j}.npy')

        sample = np.load(read_sample, allow_pickle=True)
        label = np.load(read_label, allow_pickle=True).squeeze()

        x_train, y_train = get_train_x_y(sample, label)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

        model = AdaBoostClassifier()
        model.fit(x_train, y_train.astype('int'))
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        model_dir = save_metrics_and_probs("AdaBoost", save_root, j,
                                           y_test, y_pred, y_prob, metrics)

    finalize_metrics("AdaBoost", model_dir, metrics)





if __name__ == "__main__":


    # importance_feature()
    # data()
    # func_C()
    # func_R()
    # func_XGB()
    # func_GBDT()
    # func_RandomForest()
    # func_AdaBoost()
    # GBDT()
    # XGBoost()
    # RandomForest()
    # AdaBoost()

