import librosa
import soundfile as sf
import numpy as np
import os
import torch

# from sklearn.model_selection import train_test_split
import torch.utils.data as data
from torch import nn, optim

from TCNN import TCNN

'''
# 将音频数据转化为numpy数组
'''


def preprocess_data(audio_file_path):
    # 设置参数
    sr = 16000
    duration = 2

    # 初始化变量
    data = []  # 存放音频数据

    # 加载音频文件
    filepath = audio_file_path
    y, sr = sf.read(filepath)

    # 转化为单声道
    if len(y.shape) > 1:
        y = librosa.to_mono(y)

    # 修改为16kHz采样率
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # 归一化
    y = librosa.util.normalize(y)  # 归一化

    # 剪切或补 0
    if len(y) < 32000:
        y = np.pad(y, (32000 - len(y), 0), mode='constant')
    else:
        y = y[0:32000]

    # 将数据添加到列表中，包含标签和数据
    data.append(y.reshape(1, -1))  # 将数据转化为 1 行，-1 列的形式

    # 转化为 numpy 数组
    data = np.array(data)

    return data


'''
# 建立音频路径-标签之关系
'''


def path2target(audio_dir):
    # 初始化变量
    filepath_array = []  # 存放音频数据
    labels = []

    # 得到音频文件列表
    files = os.listdir(audio_dir)

    # 处理每个音频文件
    for file in files:
        # 加载音频文件
        filepath = os.path.join(audio_dir, file)
        audios = os.listdir(filepath)
        for audio in audios:
            audiofilepath = os.path.join(filepath, audio)
            filepath_array.append(audiofilepath)
            if 'negative' in audiofilepath:
                labels.append(0)
            else:
                labels.append(1)

    return filepath_array, labels


# 建立文件路径与标签的对应关系

# windows 下训练时取消下面一行的注视
# filepath_array, labels = path2target(r'D:\大三下文件夹\大创\DataSet\2sData_bak')
# ubuntu 下训练时取消下面一行的注视
# filepath_array, labels = path2target(r'/media/renjixiong/Data/大三下文件夹/大创/DataSet/2sData_bak')
# WSL 下训练时取消下面一行注释
filepath_array, labels = path2target(r'/home/renjixiong/Model_Data/DataSet/2sData_bak')

# print(len(labels))

'''
# 划分测试集、训练集、验证集
'''

import random

# 将数据路径和对应标签打包成元组列表
data_pairs = list(zip(filepath_array, labels))

# 随机打乱元组列表
random.shuffle(data_pairs)

# 计算每个字符串列表应包含的元素数量
total_count = len(data_pairs)
list1_count = int(total_count * 0.6)
list2_count = int(total_count * 0.2)

# 使用切片操作将元组列表分割成3个比例为6:2:2的元组列表
lists = [data_pairs[:list1_count], data_pairs[list1_count:list1_count + list2_count],
         data_pairs[list1_count + list2_count:]]

# 分别提取字符串列表和数字列表
path_lists = [[pair[0] for pair in sublist] for sublist in lists]
label_lists = [[pair[1] for pair in sublist] for sublist in lists]

# 定义训练集、验证集、测试集数据容器
train_data = []
train_label = []
val_data = []
val_label = []
test_data = []
test_label = []
# 保存结果
for i, path_list in enumerate(path_lists):
    if i == 0:
        train_data = path_lists[i]
        # print(train_data)
        train_label = label_lists[i]
    if i == 1:
        val_data = path_lists[i]
        val_label = label_lists[i]
    if i == 2:
        test_data = path_lists[i]
        test_label = label_lists[i]

'''
# 生成DataLoader
'''

from torch.utils.data import Dataset, DataLoader

batch_size = 64


class MyDataset(Dataset):
    def __init__(self, data_path, data_label):
        self.data_path = data_path
        self.data_label = data_label

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        # 加载数据
        data = self.data_path[idx]
        # print(data)
        # 进行预处理
        data = preprocess_data(data)
        # 获取标签
        label = self.data_label[idx]
        # 转换为 tensor
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()
        return data, label


# 训练集
train_dataset = MyDataset(train_data, train_label)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 验证集
val_dataset = MyDataset(val_data, val_label)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 测试集
test_dataset = MyDataset(val_data, val_label)
test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# print(train_dataset)

'''
# 训练与验证函数的定义
'''


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_loss_per_batch = 0.0
        train_acc = 0.0
        train_acc_per_batch = 0.0
        val_loss = 0.0
        val_acc = 0.0
        model.train()
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            # print(type(outputs))
            # print(type(label))
            label_loss = label.to(torch.int64)
            loss = criterion(outputs, label_loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_loss_per_batch += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds == label.data)
            train_acc_per_batch += torch.sum(preds == label.data)
            #if (i + 1) % batch_size == 0:
            # print('         Epoch [{}/{}], Batch [{}], train_loss: {:.4f} %, train_acc: {:.4f} %'.format(
            #         epoch + 1, num_epochs, i, train_loss_per_batch, train_acc_per_batch))
            # train_acc_per_batch = 0.0
            # train_loss_per_batch = 0.0

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        for i, (data, label) in enumerate(val_loader):
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            label_loss = label.to(torch.int64)
            loss = criterion(outputs, label_loss)
            val_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == label.data)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        print('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))


def test(model, test_loader, criterion):
    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            label_loss = label.to(torch.int64)
            loss = criterion(outputs, label_loss)
            test_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds == label.data)

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)

        print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(test_loss, test_acc))


"""
# 定义超参数
"""
# 加载模型
model = TCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=20)

# 验证
test(model, test_dataloader, criterion)
