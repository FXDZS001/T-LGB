import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
from dataloader import Dataset, Load_Dataset_Test
from ViT_model import ViT
import matplotlib.pyplot as plt
from sklearn import manifold
from einops import rearrange
import os
if __name__ == "__main__":

    # Select the specified path
    save_path = r'D:\hainan'   # 数据集路径


    feature, label = Load_Dataset_Test(save_path)
    _, _, channels, sampling_points = feature.shape
    test_set = Dataset(feature, label, transform=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=feature.shape[0], shuffle=False)
# -------------------------------------------------------------------------------------------------------------------- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = ViT(n_class=4, sampling_point=sampling_points, dim=64, depth=6,
              heads=8, mlp_dim=64, dropout=0, emb_dropout=0).to(device)  # ec为35通道输入

    weight_path = 'D:/hainan/' + 'best model/' + str(21) + '/test_max_acc.pt'
    net.load_state_dict(torch.load(weight_path))
# -------------------------------------------------------------------------------------------------------------------- #
    net.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ec, outputs = net(inputs)

            # EC_ViT = np.array(EC_ViT)
            print(ec.shape)
            ec = ec.cpu().numpy()
            np.save(os.path.join(save_path, '回归/', 'feature/', 'feature21' + '.npy'), ec)


