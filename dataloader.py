import torch
import pandas as pd
import numpy as np
import scipy.io as scio
from scipy import signal
import os

def Load_Dataset_Train(data_path):
    """

    Args:
        data_path (str): dataset path.
        model (str): 'fNIRS-T'. fNIRS-PreT uses raw data.

    Returns:
        feature : Train fNIRS signal data.
        label : Train fNIRS labels.
    """
    feature_path = os.path.join(data_path, 'ec高空\数据', 'TiVec_samples1' + '.npy')
    label_path = os.path.join(data_path,'ec高空\ViT_label\总站\低\label', 'label1' + '.npy')

    Train_feature = np.load(feature_path, allow_pickle=True)
    Train_label = np.load(label_path, allow_pickle=True)

    feature = Train_feature
    label = Train_label
    label = label.squeeze()
    feature = np.array(feature, dtype='float32')
    label = np.array(label)
    print(feature.shape)


    return feature, label


# 同上
def Load_Dataset_Test(data_path):
    """

    Args:
        data_path (str): dataset path.
        sub : subject num.
        num : paradigm num.

    Returns:
        feature : Test fNIRS signal data.
        label : Test fNIRS labels.
    """
    feature_path = os.path.join(data_path, 'ec高空\数据', 'TiVec_samples1' + '.npy')

    label_path = os.path.join(data_path, 'ec高空\ViT_label\总站\低\label', 'label1' + '.npy')

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
    Load data for training

    Args:
        feature: input data.
        label: class for input data.
        transform: Z-score normalization is used to accelerate convergence (default:True).
    """
    # 将输入的feature, label转换成张量，并赋予当前类的实例的对应属性，
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.classes = ['Class 1', 'Class 2']  # 添加类别信息
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)# 当前类的实例的 feature 属性
        self.label = torch.tensor(self.label, dtype=torch.float)
        # print(self.feature.shape)
        # print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # z-score normalization
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()# 计算平均值和标准差，用于 Z 分数规范化
            self.feature[item] = (self.feature[item] - mean) / std # Z 分数规范化

        return self.feature[item], self.label[item]

def Load_Dataset_Train_2(data_path):
    """

    Args:
        data_path (str): dataset path.
        model (str): 'fNIRS-T'. fNIRS-PreT uses raw data.

    Returns:
        feature : Train fNIRS signal data.
        label : Train fNIRS labels.
    """
    feature_path = os.path.join(data_path, 'data/ec高空/数据', 'TiVec_samples7' + '.npy')
    label_path = os.path.join(data_path, 'data/ec高空/ViT_label/总站/低/label', 'label7' + '.npy')

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




