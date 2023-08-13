"""
此文件用于从pytorch上下载官方的示例程序
用于猫狗训练
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime

from 模型和网络 import Covnet_3

cudnn.benchmark = True
plt.ion()  # interactive mode

data_dir = r"D:\学习\大创\data\训练数据集\data\logMel 1：1(new)_2023-02-09-15-50-34"
# data_dir = r"C:\Users\28101\Desktop\大创测试数据\test"
epochs_num = 32
bath_size = 32
learning_rate = 0.001

work_path = r"D:\学习\大创\data\训练数据集\model"
# negative = 'cat'
# positive = 'dog'
negative = 'negative'
positive = "positive"
train_negative_path = data_dir + "\\" + "train\\" + negative
train_positive_path = data_dir + "\\" + "train\\" + positive
val_negative_path = data_dir + "\\" + "val\\" + negative
val_positive_path = data_dir + "\\" + "val\\" + positive
all_photo_num = len(os.listdir(train_positive_path)) + len(os.listdir(train_negative_path))
all_photo_num += len(os.listdir(val_positive_path)) + len(os.listdir(val_negative_path))
step_num_all = int(0.8 * all_photo_num / bath_size)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

"""
加载数据
"""
"""
设置参数初始化的格式
"""


def init_weights(layer):
    """
    参数初始化设置使用
    :param layer:
    :return:
    """
    # 如果为卷积层，使用 He initialization 方法正态分布生成值，生成随机数填充张量
    if type(layer) == nn.Conv2d:
        # nn.init.normal_(layer.weight, mean=0, std=0.5)
        # kaiming_normal: kaiming 正态分布
        # nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
        # kaimin_uniform:归一化初始化
        nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.42746902, 0.6474644, 0.4993739], [0.38065454, 0.35554656, 0.3535182])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.42746902, 0.6474644, 0.4993739], [0.38065454, 0.35554656, 0.3535182])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=bath_size,
                                              shuffle=True, num_workers=4, drop_last=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

"""
定义一个模型训练的函数
"""


def train_model(model, criterion, optimizer, scheduler, num_epochs=epochs_num):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.apply(init_weights)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            step_num = 0
            for inputs, labels in dataloaders[phase]:
                step_num += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 打印一下每一步的损失和准确率
                print("共{}步,第{}步  loss:{}  acc:{}".format(step_num_all, step_num, loss.item(),
                                                              torch.sum(preds == labels.data) / bath_size))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = float(epoch_acc.cpu())
            print("acc_type:{}".format(type(epoch_acc)))

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            if phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


"""
显示几个图像预测值的通用函数
"""


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                plt.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# model_ft = models.resnet18(weights=None)
model_ft = Covnet_3.Covnet()
# 获取网络名称
model_name = model_ft.__class__.__name__
# num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
# Decay LR by a factor of 0.1 every 7 epochs
# 每7个时期将LR衰减0.1倍
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
# 正弦函数形式衰减学习率
exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=4)
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.6)

if __name__ == "__main__":
    """
    训练与评估
    """
    torch.backends.cudnn.enabled = False
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epochs_num)

    """
    打印出图片做验证
    """
    # visualize_model(model_ft)

    """
    绘图
    """
    dir_count = data_dir.rfind('\\') + 1
    dir_path = data_dir[dir_count:]
    # 判断文件夹是否存在
    photo_folder = os.path.join(work_path, 'photo', dir_path)
    cd = os.path.exists(photo_folder)
    if cd:
        print("图片保存文件夹已存在")
    else:
        print("创建图片保存文件夹")
        os.mkdir(photo_folder)

    # 加时间戳
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    # plt.ylim((0, 1))
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig(photo_folder + "\\" + model_name + "18验证网络_loss_" + str(nowTime) + ".jpg")
    # plt.show()
    # plt.xlim((0,50))
    # plt.ylim((0,1))
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.ylim((0, 1))  # 限制一下绘图的幅度，更具有代表性一些
    plt.legend(["train", "val"], loc="lower right")
    plt.savefig(photo_folder + "\\" + model_name + "18验证网络_acc_" + str(nowTime) + ".jpg")
