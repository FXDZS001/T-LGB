"""
Data loading utilities for fog-visibility classification.

Purpose:
    Provide numpy/PyTorch loaders and a Dataset wrapper with optional per-sample
    Z-score normalization.

Main inputs:
    - Paths to pre-saved feature and label files.
    - Optional flag to enable per-sample standardization.

Main outputs:
    - Numpy arrays or PyTorch Dataset objects for training and testing.
"""

import torch
import pandas as pd
import numpy as np
import scipy.io as scio
from scipy import signal
import os

def Load_Dataset_Train(data_path):
    """
    Load training features and labels from disk.

    Inputs:
        - Directory path containing the training feature/label files.

    Outputs:
        - Feature array ready for model training.
        - Label array containing binary classes (fog vs non-fog).
    """
    feature_path = os.path.join(data_path, 'ec_high', 'data', 'TiVec_samples1.npy')
    label_path = os.path.join(data_path, 'ec_high', 'ViT_label', 'main_station', 'low', 'label1.npy')

    Train_feature = np.load(feature_path, allow_pickle=True)
    Train_label = np.load(label_path, allow_pickle=True)

    feature = Train_feature
    label = Train_label
    label = label.squeeze()
    feature = np.array(feature, dtype='float32')
    label = np.array(label)
    print(feature.shape)


    return feature, label


def Load_Dataset_Test(data_path):
    """
    Load test features and labels from disk.

    Inputs:
        - Directory path containing the test feature/label files.

    Outputs:
        - Feature array for evaluation or inference.
        - Label array aligned with the loaded features.
    """
    feature_path = os.path.join(data_path, 'ec_high', 'data', 'TiVec_samples1.npy')

    label_path = os.path.join(data_path, 'ec_high', 'ViT_label', 'main_station', 'low', 'label1.npy')

    Train_feature = np.load(feature_path)
    Train_label = np.load(label_path)
    feature = Train_feature
    label = Train_label
    label = label.squeeze()
    feature = np.array(feature, dtype='float32')
    label = np.array(label)
    print(label.shape)

    return feature, label


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for visibility classification.

    Inputs:
        - Feature array.
        - Label array (binary classes).
        - Optional flag to standardize each sample.

    Outputs:
        - On indexing, returns a single example (feature, label) pair ready for a model.
    """
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.classes = ['Class 1', 'Class 2'] 
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)
        # print(self.feature.shape)
        # print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std 

        return self.feature[item], self.label[item]

def Load_Dataset_Train_2(data_path):
    """
    Create and return a PyTorch Dataset for training with standardization enabled.

    Inputs:
        - Root path to training data files.

    Outputs:
        - A Dataset object that yields (feature, label) pairs for training.
    """
    feature_path = os.path.join(data_path, 'data', 'ec_high', 'data', 'TiVec_samples7.npy')
    label_path = os.path.join(data_path, 'data', 'ec_high', 'ViT_label', 'main_station', 'low', 'label7.npy')

    Train_feature = np.load(feature_path, allow_pickle=True)
    Train_label = np.load(label_path, allow_pickle=True)

    feature = Train_feature
    label = Train_label
    label = label.squeeze()
    feature = np.array(feature, dtype='float32')
    label = np.array(label)
    print(feature.shape)
    train_set = Dataset(feature, label, transform=True)  

    return train_set

class NpyDataset(Dataset):
    """
    Minimal Dataset that loads features and labels from numpy files.

    Inputs:
        - Paths to feature and label files.
        - Optional transform applied to each feature example.

    Outputs:
        - (feature, label) pairs compatible with PyTorch DataLoader.
    """
    def __init__(self, data_path, label_path, transform=None):
        self.data_path = data_path  
        self.label_path = label_path  
        self.transform = transform
        self.data = np.load(self.data_path)
        self.labels = np.load(self.label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image) 

        return image, label




