"""
3D t-SNE comparison across input sources.

Purpose:
    Compare class separability of features derived from different sources
    (e.g., surface observations, near-surface predictors, upper-air predictors)
    using three-dimensional t-SNE.

Inputs:
    - Feature matrices and labels per source.
    - Visualization parameters and optional preprocessing steps.

Outputs:
    - Side-by-side 3D scatter plots for visual comparison.
"""
import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader1 import Dataset, Load_Dataset_Train
from ViT_model import ViT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D


data_dir   = r"D:\fog_prediction\data\ec_high\data"      
label_dir  = r"D:\fog_prediction\data\ec_high\ViT_label\main_station\low\label"  
weight_dir = r"D:\fog_prediction\best model"             


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_files   = [f"TiVec_samples{i}.npy" for i in range(1, 22)]
label_files  = [f"label{i}.npy" for i in range(1, 22)]
weight_files = [f"{weight_dir}/{i}/test_max_acc.pt" for i in range(1, 22)]

fig = plt.figure(figsize=(22, 12))
gs = fig.add_gridspec(
    nrows=4, ncols=7,
    height_ratios=[0.18, 1, 1, 1],  
    left=0.03, right=0.99, top=0.97, bottom=0.05,
    wspace=0.10, hspace=0.20
)

legend_ax = fig.add_subplot(gs[0, :])
legend_ax.axis("off")

axes = [fig.add_subplot(gs[1 + r, c]) for r in range(3) for c in range(7)]

colors = ['red', 'green']  

pbar = tqdm(total=len(data_files), desc="Extracting Features", unit="file")

hours = [f"{i*3} hours" for i in range(1, 22)]

for idx, (data_file, label_file, weight_file) in enumerate(zip(data_files, label_files, weight_files)):
    data_path  = os.path.join(data_dir, data_file)
    label_path = os.path.join(label_dir, label_file)

    feature, label = Load_Dataset_Train(data_path, label_path)
    _, _, channels, sampling_points = feature.shape
    train_data, test_data, train_label, test_label = \
        train_test_split(feature, label, test_size=0.8, random_state=1)

    test_set   = Dataset(test_data, test_label, transform=True)
    test_tensor = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    net = ViT(n_class=2, sampling_point=sampling_points, dim=64,
              depth=6, heads=8, mlp_dim=64, dropout=0, emb_dropout=0).to(device)
    net.load_state_dict(torch.load(weight_file))
    net.eval()


    features, labels = [], []
    with torch.no_grad():
        for data in test_tensor:
            inputs, label_batch = data
            inputs = inputs.to(device)
            label_batch = label_batch.to(device)

            feature_patch = net(inputs)  
            features.append(feature_patch.cpu().numpy())
            labels.append(label_batch.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels   = np.concatenate(labels, axis=0).astype(int)

    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]
    class_1_downsampled = resample(
        class_1_features, replace=False, n_samples=len(class_0_features), random_state=42
    )
    features_balanced = np.vstack((class_0_features, class_1_downsampled))
    labels_balanced   = np.hstack((np.zeros(len(class_0_features)), np.ones(len(class_1_downsampled)))).astype(int)

    tsne = TSNE(n_components=2, random_state=501, perplexity=50, learning_rate=200, max_iter=2000)
    features_2d_balanced = tsne.fit_transform(features_balanced)

    ax = axes[idx]
    for i in range(features_2d_balanced.shape[0]):
        ax.scatter(features_2d_balanced[i, 0], features_2d_balanced[i, 1],
                   c=colors[labels_balanced[i]], s=10, alpha=0.7)
    ax.set_title(hours[idx], fontsize=10)
    ax.axis("off")

    pbar.update(1)

pbar.close()


handles = [
    Line2D([0],[0], marker='o', color='white', markerfacecolor='green',
           label='Non-fog (≥1000 m)', markersize=10),
    Line2D([0],[0], marker='o', color='white', markerfacecolor='red',
           label='Fog (<1000 m)', markersize=10),
]
legend_ax.legend(handles=handles, loc='center left', frameon=True,
                 facecolor='white', edgecolor='black', fontsize=12,
                 bbox_to_anchor=(0.01, 0.5))

plt.tight_layout(rect=[0, 0.04, 1, 1])
out_path = "tsne_features_hours.png"
plt.savefig(out_path, dpi=300)
plt.show()
print(f"[SAVE] t-SNE save to: {out_path}")
