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
    save_path = r'D:\hainan'  # 数据集路径
    feature, label = Load_Dataset_Test(save_path)
    _, _, channels, sampling_points = feature.shape
    test_set = Dataset(feature, label, transform=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=feature.shape[0], shuffle=False)
# -------------------------------------------------------------------------------------------------------------------- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = ViT(n_class=2, sampling_point=sampling_points, dim=64, depth=6,
              heads=8, mlp_dim=64, dropout=0, emb_dropout=0).to(device)  # ec为35通道输入

    weight_path = 'D:/hainan/' + 'best model/' + str(1) + '/test_max_acc.pt'
    net.load_state_dict(torch.load(weight_path))
# -------------------------------------------------------------------------------------------------------------------- #
    net.eval()


    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # feature_patch, outputs = net(inputs)
            feature_patch = net(inputs)
            # pred = outputs.argmax(dim=1, keepdim=True)

            feature_patch_np = feature_patch.cpu().numpy()
            label = labels.cpu().numpy()



            '''t-SNE'''
            tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='random', random_state=501)
            outputs_np = feature_patch.cpu().numpy()
            X_tsne = tsne.fit_transform(outputs_np)
            y = labels.cpu().numpy()

            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                         fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.show()

            '''t-SNE'''
            tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='random', random_state=501)
            outputs_np = feature_patch.cpu().numpy()
            X_tsne = tsne.fit_transform(outputs_np)
            y = labels.cpu().numpy()

            '''
            特征可视化
            '''
            feature_patch_np_all = np.array(feature_patch_np)

            label_all = np.array(label)
            # print(feature_patch_np_all.shape)
            # feature_patch_all = rearrange(feature_patch_np_all, 'b c   -> (b c) ')
            # label_all = rearrange(label_all, 'b  -> b ')

            '''t-SNE-3d'''
            tsne = manifold.TSNE(n_components=3, learning_rate='auto', init='random', random_state=501)
            feature_patch_tsne = tsne.fit_transform(feature_patch_np_all)
            y = label_all

            '''嵌入空间可视化'''
            feature_patch_min, feature_patch_max = feature_patch_tsne.min(0), feature_patch_tsne.max(0)
            feature_patch_norm = (feature_patch_tsne - feature_patch_min) / (feature_patch_max - feature_patch_min)  # 归一化
            plt.figure(figsize=(8, 8))
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                         fontdict={'weight': 'bold', 'size': 9})

                fig = plt.figure(figsize=(8, 8))  # 指定图像的宽和高
                plt.suptitle("Visualization of Full connection layer", fontsize=14)  # 自定义图像名称
                ax = fig.add_subplot(1, 1, 1, projection='3d')  # 创建子图
                for i in range(feature_patch_norm.shape[0]):
                    if y[i] == 1:
                        color = 'r'
                    else:
                        color = 'g'
                    ax.scatter(feature_patch_norm[i, 0], feature_patch_norm[i, 1], feature_patch_norm[i, 2], c=color, cmap=plt.cm.Spectral)  # 绘制散点图，为不同标签的点赋予不同的颜色
                ax.set_title('Original S-Curve', fontsize=14)
                ax.view_init(4, -72)  # 初始化视角
                plt.legend(labels=["Rest", "MI"], loc="upper right", fontsize=20)

                plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 删除网格线
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.gca().zaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0, 0)
                plt.xticks([])
                plt.yticks([])
                plt.show()

            '''t-SNE-2d'''
            tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='random', random_state=501)
            feature_patch_tsne = tsne.fit_transform(feature_patch_np_all)
            y = label_all

            '''patch嵌入空间可视化'''
            feature_patch_min, feature_patch_max = feature_patch_tsne.min(0), feature_patch_tsne.max(0)
            feature_patch_norm = (feature_patch_tsne - feature_patch_min) / (feature_patch_max - feature_patch_min)  # 归一化

            fig.suptitle("Visualization of patch Full connection layer", fontsize=14)  # 自定义图像名称
            ax = fig.add_subplot(1, 1, 1)
            for i in range(feature_patch_norm.shape[0]):
                if y[i] == 1:
                    color = 'r'
                else:
                    color = 'g'
                ax.scatter(feature_patch_norm[i, 0], feature_patch_norm[i, 1], c=color, cmap=plt.cm.Spectral)  # 绘制散点图，为不同标签的点赋予不同的颜色
            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 删除网格线
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=0.93, bottom=0.02, right=0.98, left=0.02, hspace=0.1, wspace = 0.05)
            plt.margins(0, 0)
            plt.xticks([])
            plt.yticks([])
            plt.title('title', y=-0.02)
            plt.legend(labels=[">=1000m", "<1000m"], loc="best", fontsize=8)

                # '''channel嵌入空间可视化'''
                # feature_channel_min, feature_channel_max = feature_channel_tsne.min(0), feature_channel_tsne.max(0)
                # feature_channel_norm = (feature_channel_tsne - feature_channel_min) / (feature_channel_max - feature_channel_min)  # 归一化
                #
                # fig.suptitle("Visualization of channel Full connection layer", fontsize=14)  # 自定义图像名称
                # ax = fig.add_subplot(2, 2, num)
                # for i in range(feature_channel_norm.shape[0]):
                #     if y[i] == 1:
                #         color = 'r'
                #     else:
                #         color = 'g'
                #     ax.scatter(feature_channel_norm[i, 0], feature_channel_norm[i, 1], c=color,
                #                 cmap=plt.cm.Spectral)  # 绘制散点图，为不同标签的点赋予不同的颜色
                # # plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 删除网格线
                # # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.subplots_adjust(top=0.93, bottom=0.02, right=0.98, left=0.02, hspace=0.1, wspace=0.05)
                # plt.margins(0, 0)
                # plt.xticks([])
                # plt.yticks([])
                # plt.title(fig_name[num-1], y=-0.08)
                # plt.legend(labels=["Rest", "MI"], loc="best", fontsize=8)

                # '''cat嵌入空间可视化'''
                # feature_cat_min, feature_cat_max = feature_cat_tsne.min(0), feature_cat_tsne.max(0)
                # feature_cat_norm = (feature_cat_tsne - feature_cat_min) / (feature_cat_max - feature_cat_min)  # 归一化
                #
                # fig.suptitle("Visualization of cat Full connection layer", fontsize=14)  # 自定义图像名称
                # ax = fig.add_subplot(2, 2, num)
                # for i in range(feature_cat_norm.shape[0]):
                #     if y[i] == 1:
                #         color = 'r'
                #     else:
                #         color = 'g'
                #     ax.scatter(feature_cat_norm[i, 0], feature_cat_norm[i, 1], c=color,
                #                 cmap=plt.cm.Spectral)  # 绘制散点图，为不同标签的点赋予不同的颜色
                # # plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 删除网格线
                # # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.subplots_adjust(top=0.93, bottom=0.04, right=0.98, left=0.02, hspace=0.1, wspace=0.05)
                # plt.margins(0, 0)
                # plt.xticks([])
                # plt.yticks([])
                # plt.title(fig_name[num-1], y=-0.08)
                # plt.legend(labels=["Rest", "MI"], loc="best", fontsize=8)
                #
            plt.show()







