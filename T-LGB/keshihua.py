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
def rotate_points(points, angle):
    """旋转二维数据点集合"""
    theta = np.radians(angle)  # 将角度转换为弧度
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    return np.dot(points, rotation_matrix.T)
# 路径设置
data_dir = "D:\\fog_prediction\\data\\ec高空\\数据"  # 输入数据存放路径
label_dir = "D:\\fog_prediction\\data\\ec高空\\ViT_label\\总站\\低\\label"  # 标签存放路径
weight_dir = "D:\\fog_prediction\\best model"  # 模型权重存放路径

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据文件、标签文件和权重文件的名称列表
data_files = [f"TiVec_samples{i}.npy" for i in range(1, 22)]  # 数据文件名
label_files = [f"label{i}.npy" for i in range(1, 22)]  # 标签文件名
weight_files = [f"{weight_dir}/{i}/test_max_acc.pt" for i in range(1, 22)]  # 模型权重文件路径

# 创建多子图（3行7列）
fig, axes = plt.subplots(3, 7, figsize=(20, 10))  # 3行7列布局
axes = axes.ravel()  # 将axes展平成一维数组，方便循环操作

# 自定义颜色映射
colors = ['red', 'green']  # 类别 0 和 1 的颜色

# 进度条
pbar = tqdm(total=len(data_files), desc="Extracting Features", unit="file")

for idx, (data_file, label_file, weight_file) in enumerate(zip(data_files, label_files, weight_files)):
    # 加载数据和标签路径
    data_path = os.path.join(data_dir, data_file)
    label_path = os.path.join(label_dir, label_file)

    # 使用 Load_Dataset_Train 加载数据和标签
    feature, label = Load_Dataset_Train(data_path, label_path)
    _, _, channels, sampling_points = feature.shape
    train_data, test_data, train_label, test_label = \
        train_test_split(feature, label, test_size=0.8, random_state=1)

    test_set = Dataset(test_data, test_label, transform=True)  # 确保 Dataset 的定义支持 transform
    test_tensor = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    print(data_file,label_file)
    # 加载模型权重
    net = ViT(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64, dropout=0, emb_dropout=0).to(device)
    net.load_state_dict(torch.load(weight_file))
    net.eval()

    # 提取特征
    features = []  # 初始化特征数组
    labels = []    # 初始化标签数组
    with torch.no_grad():
        for data in test_tensor:
            inputs, label_batch = data
            inputs = inputs.to(device)
            label_batch = label_batch.to(device)

            # 提取特征
            feature_patch = net(inputs)  # 假设这里 net 输出的是特征
            features.append(feature_patch.cpu().numpy())  # 将特征加入列表
            labels.append(label_batch.cpu().numpy())  # 将标签加入列表

    # 合并所有特征和标签
    features = np.concatenate(features, axis=0)  # 特征数组，形状为 (N, D)
    labels = np.concatenate(labels, axis=0)      # 标签数组，形状为 (N,)
    labels = labels.astype(int)
    predictions = np.argmax(features, axis=1)
    accuracy = accuracy_score(labels, predictions)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    print("Feature Mean:", np.mean(features, axis=0))
    print("Feature Std Dev:", np.std(features, axis=0))
    # 打印每个类别的样本数
    unique, counts = np.unique(labels, return_counts=True)
    print('类别数：', dict(zip(unique, counts)))

    # 获取类别 0 和类别 1 的数据
    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]

    # 对类别 1 进行下采样，使其数量等于类别 0
    class_1_downsampled = resample(class_1_features,
                                   replace=False,  # 不重复采样
                                   n_samples=len(class_0_features),  # 下采样后的样本数与类别 0 相同
                                   random_state=42)  # 设置随机种子保证可复现
    # 合并类别 0 和下采样后的类别 1
    features_balanced = np.vstack((class_0_features, class_1_downsampled))

    # 创建标签数组，类别 0 的标签为 0，类别 1 的标签为 1
    labels_balanced = np.hstack((np.zeros(len(class_0_features)), np.ones(len(class_1_downsampled))))
    labels_balanced = labels_balanced.astype(int)

    # 使用 t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=501 ,perplexity=50, learning_rate=200, max_iter=2000)
    features_2d_balanced = tsne.fit_transform(features_balanced)

    # 绘制到子图
    ax = axes[idx]
    for i in range(features_2d_balanced.shape[0]):
        ax.scatter(features_2d_balanced[i, 0], features_2d_balanced[i, 1], c=colors[labels_balanced[i]], s=10, alpha=0.7)
    ax.set_title(f"Feature {idx + 1}", fontsize=10)
    ax.axis("off")  # 隐藏坐标轴


    # 更新进度条
    pbar.update(1)

# 进度条结束
pbar.close()
# 调整布局并保存图片
plt.tight_layout()
output_path = "perplexity=50_2,random_state=501.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"t-SNE visualization saved to {output_path}")
