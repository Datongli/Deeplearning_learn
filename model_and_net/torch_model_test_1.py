"""
此文件用于高度集成化代码，能封装进函数的都封装进函数，明晰代码的逻辑
"""
"""
现在想的是，可以train和val放在一个函数下面，然后test单独的一个函数
"""
import imageio.v2 as imageio
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import itertools
import os
import random
# import cv2
import matplotlib.pyplot as plt
import numpy as np
# PaddyDataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
import GhostNet
import efficientnet
import GhostNet_res
import Covnet
import Covnet_2
import Covnet_3
import ResNet
from modeltools import labels_name, PaddyDataSet_train_val, PaddyDataSet_test, init_weights, count_train_num, \
    exists_or_mkdir, GPU_is_alivable, calculation_mean_std, train_and_val


labels_list = ['negative', 'positive']
# 标签的实际名称(现在仅限二分类)
# labels:一部字典，键为标签名称，值为0或1
labels = labels_name(labels_list)

# 工作目录
work_path = r"D:\学习\大创\data\训练数据集\model"
# 训练加验证数据集文件夹位置
filepath_train_val = r"D:\学习\大创\data\训练数据集\data\Track1+CoughVid 谱图合集\Track1+CoughVid logMel"
# 测试数据集文件夹位置
filepath_test = r"D:\学习\大创\data\训练数据集\data\Track1+CoughVid 谱图合集\TFDF(2s)"

"""
（0）参数设置
"""
batch_size = 16  # 每个step训练batch_size张图片
epochs = 2  # 共训练epochs次
k = 5  # k折交叉验证
dropout_num_1 = 0.4
dropout_num_2 = 0.4
resnet_dropout = 0.4
learning_rate = 1e-4
pre_score_k = []
labels_k = []
# wd：正则化惩罚的参数
wd = 0.2
print("wd:{}".format(wd))
# wd = None
# stop_epoch: 早停的批量数
stop_epoch = 4

"""
（1）前期准备
"""
# 需要用到train_num 初始化一下混淆矩阵
train_num = count_train_num(filepath_train_val=filepath_train_val, labels=labels)

# 检测权重和保存图片的文件夹是否存在，不存在则创建
# savepath:权重文件夹
# photo_folder:保存图片文件夹
# dir_path:数据集所在的文件夹名称
savepath, photo_folder, dir_path = exists_or_mkdir(data_path=filepath_train_val, work_path=work_path)

# 获取GPU设备
device = GPU_is_alivable()

"""
（2）构造数据集
"""
image_mean, image_std = calculation_mean_std(filepath=filepath_train_val)

# logmel 1:1
# image_mean = [0.33478397, 0.55253255, 0.5409211]
# image_std = [0.36969644, 0.3965568, 0.33030042]

# 实例化一个对象，用于承接train和val的数据的迭代器
train_val_data = PaddyDataSet_train_val(data_dir=filepath_train_val,
                                        transform=transforms.Compose([
                                            # 将输入图像大小调整为224*224
                                            transforms.Resize((224, 224)),
                                            # # 数据增强，随机水平翻转
                                            # transforms.RandomHorizontalFlip(),
                                            # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=image_mean, std=image_std)
                                        ]))
# 实例化一个承接test数据的迭代器
test_data = PaddyDataSet_test(data_dir=filepath_test,
                              transform=transforms.Compose([
                                  # 将输入图像大小调整为224*224
                                  transforms.Resize((224, 224)),
                                  # # 数据增强，随机水平翻转
                                  # transforms.RandomHorizontalFlip(),
                                  # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=image_mean, std=image_std)
                              ]))
# 获取一下测试集的序号，后续测试会用到
test_index = [i for i in range(len(test_data))]

# # 先划分成 5份
kf = KFold(n_splits=k, shuffle=True, random_state=34)
# 初始化混淆矩阵
cnf_matrix = np.zeros([2, 2])
# 计算一下训练有多少步
step_num = int(train_num / batch_size)

"""
（3）训练前期准备
"""
# 这一段代码是为了过程化训练进程做准备
line_num = step_num / 20.0
num_1 = line_num

# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

"""
（4）模型的训练
"""
# 第几折交叉验证
k_num = 0
# 五折中最高的准确率
best_acc_all = 0
for train_index, val_index in kf.split(train_val_data):
    """
    每一折都要实例化新的模型，不然模型会学到测试集的东西
    """
    # 选择使用的网络
    net = ResNet.resnet18(num_classes=2, include_top=True, dropout=resnet_dropout)
    # 给模型参数进行初始化
    # net.apply(init_weights)

    """
    定义损失函数，优化器，学习率更新策略，分发数据
    """
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wd)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16)

    # 分发数据
    train_fold = torch.utils.data.dataset.Subset(train_val_data, train_index)
    val_fold = torch.utils.data.dataset.Subset(train_val_data, val_index)
    test_fold = torch.utils.data.dataset.Subset(test_data, test_index)
    # 打包成DataLoader类型 用于 训练
    train_loader = DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    # 计算训练集,验证集,测试集的大小
    train_num = len(train_fold)
    val_num = len(val_fold)

    """
    调用训练和验证的函数
    """
    train_and_val(net=net, device=device, photo_folder=photo_folder,
                  nowTime=nowTime, batch_size=batch_size,
                  epochs=epochs, learning_rate=learning_rate,
                  train_loader=train_loader, val_loader=val_loader,
                  line_num=line_num, step_num=step_num,
                  optimizer=optimizer, loss_function=loss_function,
                  scheduler=scheduler, train_num=train_num,
                  val_num=val_num, savepath=savepath, dir_path=dir_path,
                  k_num=k_num, wd=wd, stop_epoch=stop_epoch,
                  dropout_num_1=dropout_num_1, dropout_num_2=dropout_num_2,
                  resnet_dropout=resnet_dropout)



    """
    测试集，用于判断测试的准确率以及绘制roc曲线和混淆矩阵
    """
    # 使用刚刚训练的权重
    savename = savepath + '\\model_' + dir_path + "_第{}折验证".format(k_num) + net_name + "网络" + '.pth'
    weightpath = savename
    # 初始化网络
    net = ResNet.resnet18(num_classes=2, include_top=True)
    # net = models.resnet18(weights=None)
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 2)
    # net = Covnet_2.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet.ghostnet()
    # net = Covnet_3.Covnet()
    # net = efficientnet.efficientnet_b0(num_classes=2)
    # net = Covnet.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet_res.resnet18()
    # 为网络重写分类层
    # in_channel = net.fc.in_features  # 2048
    # net.fc = nn.Linear(in_channel, 2)  # [b,2048]==>[b,2]
    # 加载权重
    net.load_state_dict(torch.load(weightpath, map_location=device))
    # 模型切换成验证模式，目的是让dropout和bn切换形式
    net.eval()
    # 将模型搬运到GPU上
    net.to(device)
    test_acc = 0.0
    pre_score = []
    labels_epoch = []
    """
    这是是模型的测试过程
    """
    with torch.no_grad():
        test_step = 0.0
        for data_test in test_loader:
            # 获取测试集的图片和标签
            test_images, test_labels = data_test
            #  前向传播
            outputs = net(test_images.to(device))
            # 添加softmax层
            # outputs = nn.Softmax(dim=1)(outputs)
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]
            # 累加每个step的准确率
            test_acc += (predict_y == test_labels.to(device)).sum().item()
            test_step += 1

            # 准备roc曲线所需要的数据
            positive_pre = outputs[:, 1]
            positive_pre = positive_pre.cpu()
            positive_pre = positive_pre.detach().numpy()
            positive_pre = positive_pre.tolist()
            labels = test_labels.detach().numpy()
            labels = labels.tolist()
            pre_score += positive_pre
            labels_epoch += labels

            # 更新混淆矩阵
            for index in range(len(test_labels)):
                cnf_matrix[predict_y[index]][labels[index]] += 1

        test_file_num = batch_size * len(test_loader)
        # 计算测试集图片的平均准确率
        acc_test = test_acc / test_file_num
        # 打印测试集的准确率
        print("第{}折测试集的acc：{}".format(k_num, acc_test))
        file.write("第{}折测试集的acc：{}\n".format(k_num, acc_test))

    # 保存k折的roc参数
    pre_score_k.append(pre_score)
    labels_k.append(labels_epoch)

    # 每一折验证的时候，都绘制loss和acc曲线
    # 加时间戳
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    # plt.ylim((0, 1))
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig(photo_folder + "\\" + net_name + "网络 model_loss_第{}折_".format(k_num) + str(nowTime) + ".jpg")
    # plt.show()
    # plt.xlim((0,50))
    # plt.ylim((0,1))
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    # plt.ylim((0, 1))  # 限制一下绘图的幅度，更具有代表性一些
    plt.legend(["train", "val"], loc="lower right")
    plt.savefig(photo_folder + "\\" + net_name + "网络 model_acc_第{}折_".format(k_num) + str(nowTime) + ".jpg")

"""
k折交叉验证的话，在前面绘制了loss和acc
绘制混淆矩阵以及每一折的ROC曲线并取平均，计算每一折AUC并取平均
"""
file.close()
# 以下是用于绘制ROC曲线的代码部分
# # 以下是用于绘制ROC曲线的代码部分
avg_x = []
avg_y = []
sum = 0
clr_1 = 'tab:green'
clr_2 = 'tab:green'
clr_3 = 'k'

plt.figure()
for i in range(k):
    fpr, tpr, thersholds = roc_curve(labels_k[i], pre_score_k[i])
    avg_x.append(sorted(random.sample(list(fpr), len(list(fpr)))))
    avg_y.append(sorted(random.sample(list(tpr), len(list(tpr)))))
    roc_auc1 = auc(fpr, tpr)

    roc_auc = roc_auc1 * 100
    sum = sum + roc_auc
    plt.plot(fpr, tpr, label='V-' + str(i + 1) + ' (auc = {0:.2f})'.format(roc_auc), c=clr_1, alpha=0.2)

data_x = np.array(avg_x, dtype=object)
data_y = np.array(avg_y, dtype=object)
avg = sum / k

# 准备数据
data_x_plt = []

data_x_num = len(data_x[0])
if data_x_num >= len(data_x[1]):
    data_x_num = len(data_x[1])
if data_x_num >= len(data_x[2]):
    data_x_num = len(data_x[2])
if data_x_num >= len(data_x[3]):
    data_x_num = len(data_x[3])
if data_x_num >= len(data_x[4]):
    data_x_num = len(data_x[4])

for i in range(5):
    data_x[i] = sorted(random.sample(data_x[i], data_x_num))

for i in range(data_x_num):
    a = 0.0
    a += data_x[0][i]
    a += data_x[1][i]
    a += data_x[2][i]
    a += data_x[3][i]
    a += data_x[4][i]
    a = a / k
    data_x_plt.append(a)

data_y_plt = []
data_y_num = len(data_y[0])
if data_y_num >= len(data_y[1]):
    data_y_num = len(data_y[1])
if data_y_num >= len(data_y[2]):
    data_y_num = len(data_y[2])
if data_y_num >= len(data_y[3]):
    data_y_num = len(data_y[3])
if data_y_num >= len(data_y[4]):
    data_y_num = len(data_y[4])

for i in range(5):
    data_y[i] = sorted(random.sample(data_y[i], data_y_num))

for i in range(data_y_num):
    a = 0.0
    a += data_y[0][i]
    a += data_y[1][i]
    a += data_y[2][i]
    a += data_y[3][i]
    a += data_y[4][i]
    a = a / k
    data_y_plt.append(a)

plt.plot(data_x_plt, data_y_plt, label='AVG (auc = {0:.2f})'.format(avg), c=clr_2, alpha=1, linewidth=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
plt.legend(loc='lower right', frameon=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(photo_folder + "\\" + net_name + "网络 model_ROC_" + str(nowTime) + ".jpg")
plt.show()

"""
绘制混淆矩阵，并保存
"""
Confusion_matrix_path = photo_folder + "\\" + net_name + "网络 Confusion matrix" + str(nowTime) + ".jpg"


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          path=Confusion_matrix_path):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #         print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    #         print(cm)
    #     else:
    #         print('显示具体数字：')
    #         print(cm)
    plt.figure(dpi=320, figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig(path)


# 第一种情况：显示百分比
# classes = ['cat', 'dog']
classes = ['negative', 'positive']
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')

# # 第二种情况：显示数字
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')
