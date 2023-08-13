"""
此文件用于绘制模型最后一层全连接的umap图
男女阴阳
"""

import torch
import umap
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from model_and_net import ResNet
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch import nn
import os
from tqdm import tqdm
import matplotlib.patches as mpatches

negative = 'negative'
positive = 'positive'
# negative = 'cat'
# positive = 'dog'
batch_size = 8
# paddy_labels = {negative: 0,
#                 positive: 1}
paddy_labels = {"n_female": 0,
                "p_female": 1,
                "n_male": 2,
                "p_male": 3}

data_dir = r"D:\学习\大创\data\训练数据集\data\Track1+CoughVid 谱图合集\测试集(over 4s)\Track1+CoughVid logMel test"
csv_path = r"C:\Users\ldt20\Desktop\Track-1+CoughVid 男女阴阳统计.csv"


"""
读取uid和阴性阳性和性别
"""
csv_data = {}
# df = pd.read_csv(csv_path)
file = open(csv_path, "r")
df = file.read()
# print(df)
print(type(df))
csv_list = df.split("\n")[1:-1]
print(csv_list)
file.close()

for row in csv_list:
    words = row.split(",")
    uid = words[0]
    if words[1] == 'n' and words[2][0] == 'f':
        label = 0
    if words[1] == 'p' and words[2][0] == 'f':
        label = 1
    if words[1] == 'n' and words[2][0] == 'm':
        label = 2
    if words[1] == 'p' and words[2][0] == 'm':
        label = 3
    csv_data[uid] = label

print(csv_data)


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
                    move_out = img_name.split('-')[-1]
                    name_middle = img_name.replace(move_out, '')
                    uid = name_middle[:-11]
                    label = csv_data[uid]
                    data_info.append((path_img, label))
        return data_info


image_mean = [0.33067024, 0.5446649, 0.540241]
image_std = [0.36937192, 0.39847413, 0.32845193]
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
    torch.load(r"C:\Users\ldt20\Desktop\训练权重保存\model_Track1+CoughVid logMel train_val最好的权重ResNet18网络.pth",
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
        output = nn.Softmax(dim=1)(output)
        features.append(output.cpu().numpy())
        labels.append(label.cpu().numpy())

# 拼接
features = np.vstack(features)
labels = np.concatenate(labels)

# 降维并绘制特征图
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
embedding = reducer.fit_transform(features)
labels = labels.tolist()
print(labels)
color = []

for i in range(len(embedding[:, 0])):
    if labels[i] == 0:
        color += ["red"]
    if labels[i] == 1:
        color += ['blue']
    if labels[i] == 2:
        color += ['green']
    if labels[i] == 3:
        color += ['yellow']



red_patch = mpatches.Patch(color='red', label="n_female")
blue_patch = mpatches.Patch(color='blue', label="p_male")
green_patch = mpatches.Patch(color='green', label="n_male")
yellow_patch = mpatches.Patch(color='yellow', label="p_male")
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, alpha=0.3)
# 调用plt.legend()显示图例，并给handles参数传入图例标签列表
plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
# plt.legend(handles=[green_patch, yellow_patch])
plt.show()
print("over")
