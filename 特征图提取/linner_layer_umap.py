"""
此文件用于绘制模型最后一层全连接的umap图
"""

import torch
import umap
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from 模型和网络 import ResNet
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
from tqdm import tqdm

negative = 'negative'
positive = 'positive'
# negative = 'cat'
# positive = 'dog'
batch_size = 8
paddy_labels = {negative: 0,
                positive: 1}
data_dir = r"C:\Users\ldt20\Desktop\small"


class PaddyDataSet_test(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        数据集
        """
        self.label_name = {negative: 0, positive: 1}
        # data_info 存储所有图片路径和标签, 在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.temp = np.zeros((224, 224))

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        # print(img.size)
        if img.size == self.temp.shape:
            img = img.resize((224, 224))
            # print(img.size)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # print(sub_dir)
                    label = paddy_labels[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info


image_mean = [0.33478397, 0.55253255, 0.5409211]
image_std = [0.36969644, 0.3965568, 0.33030042]
# 加载数据集
test_data = PaddyDataSet_test(data_dir=data_dir,
                              transform=transforms.Compose([
                                  # 将输入图像大小调整为224*224
                                  transforms.Resize((224, 224)),
                                  # # 数据增强，随机水平翻转
                                  # transforms.RandomHorizontalFlip(),
                                  # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=image_mean, std=image_std)
                              ]))

testloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# 加载模型并提取特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet.resnet18(num_classes=2, include_top=True)  # 将 MyModel 替换为你自己的模型
# model = Covnet_3.Covnet()
model.load_state_dict(
    torch.load(r"D:\学习\大创\data\训练数据集\model\pth\Track1+CoughVid logMel\model_Track1+CoughVid logMel最好的权重ResNet网络.pth",
               map_location=device))
model.to(device)
model.eval()

features = []
labels = []

with torch.no_grad():
    for data, label in tqdm(testloader, desc='test_data', unit='photo'):
        # for data, label in testloader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        features.append(output.cpu().numpy())
        labels.append(label.cpu().numpy())

features = np.vstack(features)
labels = np.concatenate(labels)

# 降维并绘制特征图
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
embedding = reducer.fit_transform(features)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, alpha=0.4)
plt.colorbar()
plt.show()
print("over")
