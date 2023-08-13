"""
该文件用于封装一些常用的模型训练函数，方便以后的调用
"""
"""
这是前期做过的一些打包工作等等，等有时间继续，现在我也不清楚做到哪一步了，慢慢来吧，总有重构完的一天
"""
import datetime
import itertools
import os
import torch.nn.functional as F
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


negative = 'negative'
positive = 'positive'
paddy_labels = {negative: 0,
                positive: 1}


def labels_name(pre_list):
    """
    该函数用于定义使用的标签是什么
    :param pre_list: 标签的真实名称
    :return: 标签的字典
    """
    negative = pre_list[0]
    positive = pre_list[1]
    # 返回值，字典
    paddy_labels = {
        negative: 0,
        positive: 1
    }
    return paddy_labels


class PaddyDataSet(Dataset):
    """
    并联网络输入的数据集，使用此dataset的前提是两个文件夹下的数据命名规则相同，数据量相同
    """
    def __init__(self, data_dir_1, data_dir_2, transform_1=None, transform_2=None):
        """
        数据集
        """
        self.label_name = {negative: 0, positive: 1}
        # data_info 存储所有图片路径和标签, 在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir_1)
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.temp = np.zeros((224, 224))

    def __getitem__(self, index):
        path_img_1, label = self.data_info[index]
        # 类别1的路径 path_img_1
        # 取出图片的名称 img_name
        img_name_num = path_img_1.rfind('\\') + 1
        img_name = path_img_1[img_name_num:]
        # 取出图片的类别 img_class
        img_class_list = path_img_1[:img_name_num-1]
        img_class_num = img_class_list.rfind('\\') + 1
        img_class = img_class_list[img_class_num:]
        # 得到类别二的路径 img_path_2
        path_img_2 = os.path.join(self.data_dir_2, img_class, img_name)
        img_1 = Image.open(path_img_1).convert('RGB')
        img_2 = Image.open(path_img_2).convert('RGB')
        if img_1.size == self.temp.shape:
            img_1 = img_1.resize((224, 224))
        if img_2.size == self.temp.shape:
            img_2 = img_2.resize((224, 224))
            # print(img.size)
        if self.transform_1 is not None:
            img_1 = self.transform_1(img_1)
        if self.transform_2 is not None:
            img_2 = self.transform_2(img_2)

        return img_1, img_2, label

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


class PaddyDataSet_model(Dataset):
    """
    并联网络输入的数据集，使用此dataset的前提是两个文件夹下的数据命名规则相同，数据量相同
    """
    def __init__(self, data_dir_1, data_dir_2, transform_1=None, transform_2=None):
        """
        数据集
        """
        self.label_name = {negative: 0, positive: 1}
        # data_info 存储所有图片路径和标签, 在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir_1)
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.temp = np.zeros((224, 224))

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        # 图片的路径 path_img
        # 取出uid
        uid = os.path.basename(path_img)[:-4]
        # 取出图片的类别 img_class
        classes = os.path.basename(os.path.dirname(path_img))
        # 得到音频的路径 path_audio
        path_audio = os.path.join(self.data_dir_2, classes, uid + ".wav")
        img = Image.open(path_img).convert('RGB')
        if img.size == self.temp.shape:
            img = img.resize((224, 224))
        if self.transform_1 is not None:
            img = self.transform_1(img)
        # 进行预处理
        audio = preprocess_data(path_audio)
        if self.transform_2 is not None:
            audio = self.transform_2(audio).float()

        return img, audio, label

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


class PaddyDataSet_train_val(Dataset):
    # 用于包装train和val数据的dataset迭代器，里面剔除了test数据
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


class PaddyDataSet_test(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        数据集
        """
        # data_info 存储所有图片路径和标签, 在DataLoader中通过index读取样本
        self.test_info = self.get_img_info(data_dir)
        self.transform = transform
        self.temp = np.zeros((224, 224))

    def __getitem__(self, index):
        path_img, label = self.test_info[index]
        img = Image.open(path_img).convert('RGB')
        # print(img.size)
        if img.size == self.temp.shape:
            img = img.resize((224, 224))
            # print(img.size)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.test_info)

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


def init_weights(layer):
    """
    参数初始化设置使用
    :param layer：传进来进行参数初始化的层
    :return:没有返回值
    """
    # 如果为卷积层，使用 He initialization 方法正态分布生成值，生成随机数填充张量
    if type(layer) == nn.Conv2d:
        # nn.init.normal_(layer.weight, mean=0, std=1)
        nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)


def getStat(all_data):
    """
    用于计算自己（图片）数据集的均值与方差
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)均值和方差
    """
    train_loader = torch.utils.data.DataLoader(
        all_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print(type(train_loader))
    print(len(all_data))
    all_num = len(all_data)
    num = 0
    for X, _ in train_loader:
        num += 1
        print("共{}个，第{}个".format(all_num, num))
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(all_data))
    std.div_(len(all_data))
    return list(mean.numpy()), list(std.numpy())


def count_train_num(filepath_train_val, labels):
    """
    用于计算训练集数据量的函数
    :param filepath_train_val: 用于训练和验证的数据集地址
    :param labels: 标签列表
    :return: 训练集的数据量
    """
    # 找到阴性阳性的数据文件夹
    negative_path = os.path.join(filepath_train_val, labels[0])
    positive_path = os.path.join(filepath_train_val, labels[1])
    # all_photo_num：所有图片数目
    all_photo_num = len(os.listdir(negative_path))
    all_photo_num += len(os.listdir(positive_path))
    # 计算两种样本的比例alpha = p:(n+p)
    negative_num = len(os.listdir(negative_path))
    positive_num = len(os.listdir(positive_path))
    alpha = positive_num / (positive_num + negative_num)
    print("{}占总体为:{}".format(labels[1], alpha))
    # 需要用到train_num 初始化一下混淆矩阵
    train_num = all_photo_num * 0.8
    return train_num


def exists_or_mkdir(data_path, work_path):
    """
    查看权重文件夹和保存图片文件夹是否存在，不存在即创建
    :param data_path: 数据集的路径，用于取其名称
    :param work_path: 权重和图片保存的根目录
    :return: 权重文件夹和保存图片文件夹的名称
    """
    # 显示一下文件夹的名称
    dir_path = os.path.basename(data_path)
    print(dir_path)
    # 创建权重的文件夹
    savepath = os.path.join(work_path, 'pth', dir_path)
    cd = os.path.exists(savepath)
    if cd:
        print("权重保存文件夹已存在")
    else:
        print("创建权重保存文件夹")
        os.mkdir(savepath)
    # 判断保存图片文件夹是否存在
    photo_folder = os.path.join(work_path, 'photo', dir_path)
    cd = os.path.exists(photo_folder)
    if cd:
        print("图片保存文件夹已存在")
    else:
        print("创建图片保存文件夹")
        os.mkdir(photo_folder)
    return savepath, photo_folder, dir_path


def GPU_is_alivable():
    """
    获取GPU设备
    :return: 得到的设备
    """
    if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
        device = torch.device('cuda:0')
        print('GPU')
    else:
        device = torch.device('cpu')
        print('CPU')
    return device


def calculation_mean_std(filepath):
    """
    计算数据集的均值与方差
    :param filepath: 数据集的路径
    :return: 计算得到的均值和方差
    """
    transform = transforms.Compose([transforms.ToTensor()])
    all_dataset = ImageFolder(root=filepath + '/', transform=transform)
    image_mean, image_std = getStat(all_dataset)
    print("image_mean:{}".format(image_mean))
    print("image_std:{}".format(image_std))
    return image_mean, image_std


def train_and_val(net, device, photo_folder,
                  nowTime, batch_size, epochs, learning_rate,
                  train_loader, val_loader, line_num, step_num,
                  optimizer, loss_function, scheduler,
                  train_num, val_num,
                  savepath, dir_path,
                  k_num, wd=0.1, stop_epoch=4,
                  dropout_num_1=0.4, dropout_num_2=0.4, resnet_dropout=0.2):
    """
    用于训练和测试过程的函数
    :param net: 实例化的网络模型，如resnet18,ghostnet等等
    :param device: CPU或者GPU设备
    :param photo_folder: 保存训练图片的文件夹名称
    :param nowTime: 时间戳
    :param batch_size: 每一个批量的数据数目
    :param epochs: 回带数目
    :param learning_rate: 学习率
    :param train_loader: 训练数据迭代器
    :param val_loader: 验证数据迭代器
    :param line_num: 可视化精度条长度，变量
    :param step_num: 步数
    :param optimizer: 优化器
    :param loss_function: 损失函数
    :param scheduler: 学习率更新策略
    :param train_num: 训练数据集的数据量
    :param val_num: 验证数据集的数据量
    :param savepath: 权重保存的地址
    :param dir_path: 数据集所在的文件夹名称
    :param k_num: 第几折交叉验证
    :param wd: L2正则化参数
    :param stop_epoch: 早停的批次
    :param dropout_num_1: convnet第一层dropout的参数
    :param dropout_num_2: convnet第二层dropout的参数
    :param resnet_dropout: resnet中的dropout层的参数
    :return:
    """
    # 将模型搬运到GPU上
    net.to(device)
    # 获取网络名称
    net_name = net.__class__.__name__
    # 保存每一折准确率最高的一次迭代
    best_acc = 0.0
    # 写一个txt文件用于保存超参数
    file_name = r"{}\{}网络 {}.txt".format(photo_folder, net_name, nowTime)
    file = open(file_name, 'w', encoding='utf-8')
    if os.path.exists(file_name):
        file.write("batch_size:{}\n epoch:{}\n learning_rate:{}\n".format(batch_size, epochs, learning_rate))
        file.write("weight_decay:{}\n".format(wd))
        file.write(
            "dropout_1:{}, dropout_2:{}, resnet_dropout:{}\n".format(dropout_num_1, dropout_num_2, resnet_dropout))

    # 初始化一些空白矩阵
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    pre_score = []
    labels_epoch = []
    min_val_loss = 100
    # 显示此时是第k折交叉验证
    k_num += 1
    print("-" * 30)
    print("第{}折验证".format(k_num))

    """
    训练过程
    """
    for epoch in range(epochs):
        print('-' * 30, '\n', '共', epochs, '个epoch, 第', epoch + 1, '个epoch')
        file.write('{}\n,共{}个epoch,第{}个epoch\n'.format('-' * 30, epochs, epoch + 1))
        # 将模型设置为训练模型, dropout层和BN层只在训练时起作用
        net.train()
        # 计算训练一个epoch的总损失
        running_loss = 0.0
        epoch_acc = 0.0
        num_0 = 0
        num_1 = line_num
        line = "[" + ">" + "·" * 19 + "]"

        # 每个step训练一个batch
        # enumerate：遍历，返回索引和元素
        for step, data in enumerate(train_loader):
            running_acc = 0.0
            num_0 += 1

            # data中包含图像及其对应的标签
            images, labels = data

            # 梯度清零，因为每次计算梯度是一个累加
            optimizer.zero_grad()

            # 前向传播
            # output是torch.tensor类型的数据 [batch_size, 2]
            outputs = net(images.to(device))

            # 计算预测值和真实值的交叉熵损失,交叉熵损失也叫softmax损失
            loss = loss_function(outputs, labels.to(device))

            # 计算acc
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]

            # 累加每个step的准确率
            running_acc = (predict_y == labels.to(device)).sum().item()
            epoch_acc += running_acc

            # # 梯度计算
            loss.backward()
            # 权重更新
            optimizer.step()

            # 累加每个step的损失
            running_loss += loss.item()

            # 可视化训练过程（进度条的形式）
            if num_0 <= num_1:
                line = "[" + "=" * int(num_1 / line_num - 1) + ">" + "·" * (19 - int(num_1 / line_num - 1)) + "]"
            else:
                num_1 += line_num
                line = "[" + "=" * int(num_1 / line_num - 1) + ">" + "·" * (19 - int(num_1 / line_num - 1)) + "]"
            # 打印每个step的损失和acc
            print(line, end='')
            print(f'共:{step_num} step:{step + 1} loss:{loss} acc:{running_acc / batch_size}')
            file.write("第{}折, 共:{} step:{} loss:{} acc:{}\n".format(k_num, step_num, step + 1, loss, running_acc / batch_size))

        """
        网络验证
        """
        net.eval()  # 切换为验证模型，BN和Dropout不进行参数的更新作用
        acc = 0.0  # 验证集准确率
        val_loss_run = 0.0

        with torch.no_grad():  # 下面不进行梯度计算

            val_step = 0.0
            # 每次验证一个batch
            for data_val in val_loader:
                # 获取验证集的图片和标签
                val_images, val_labels = data_val
                # print(val_labels)
                # 前向传播
                outputs = net(val_images.to(device))

                # 计算预测值和真实值的交叉熵损失
                loss = loss_function(outputs, val_labels.to(device))
                # loss = loss_function.forward(outputs, val_labels.to(device))

                # 累加每个step的损失
                val_loss_run += loss.item()

                # 预测分数的最大值
                predict_y = torch.max(outputs, dim=1)[1]

                # 累加每个step的准确率
                acc += (predict_y == val_labels.to(device)).sum().item()

                val_step += 1

            # 计算所有图片的平均准确率
            acc_val = acc / val_num
            acc_train = epoch_acc / train_num

            # 打印每个epoch的训练损失和验证准确率
            print(f'total_train_loss:{running_loss / (step + 1)}, total_train_acc:{acc_train}')
            print(f'total_val_loss:{val_loss_run / val_step}, total_val_acc:{acc_val}')
            train_loss.append(running_loss / (step + 1))
            train_acc.append(acc_train)
            val_loss.append(val_loss_run / val_step)
            val_acc.append(acc_val)
            file.write('total_train_loss:{}, total_train_acc:{}\n'.format(running_loss / (step + 1), acc_train))
            file.write('total_val_loss:{}, total_val_acc:{}\n'.format(val_loss_run / val_step, acc_val))

            # 进行早停的检查
            if val_loss[-1] <= min_val_loss:
                min_val_loss = val_loss[-1]
                epoch_num = epoch + 1
            if val_loss[-1] >= min_val_loss + 0.2:
                if (epoch + 1) - epoch_num >= stop_epoch:
                    # 保存的权重名称
                    savename = savepath + '\\model_' + dir_path + "_第{}折验证".format(
                        k_num) + net_name + "网络" + '.pth'
                    # 保存当前权重
                    torch.save(net.state_dict(), savename, _use_new_zipfile_serialization=False)
                    break

            # -------------------------------------------------- #
            # （6）权重保存
            # -------------------------------------------------- #
            # 保存每一折验证的最好权重
            if acc_val > best_acc:
                # 更新最佳的准确率
                best_acc = acc_val
                # 保存的权重名称
                savename = savepath + '\\model_' + dir_path + "_第{}折验证".format(k_num) + net_name + "网络" + '.pth'
                # 保存当前权重
                torch.save(net.state_dict(), savename, _use_new_zipfile_serialization=False)

            # 保存整个训练中的最好权重
            if acc_val > best_acc_all:
                # 更新最佳的准确率
                best_acc_all = acc_val
                # 保存的权重名称
                savename = savepath + '\\model_' + dir_path + "最好的权重" + net_name + "网络" + '.pth'
                # 保存当前权重
                torch.save(net.state_dict(), savename, _use_new_zipfile_serialization=False)
        # 学习率更新：根据回带的次数来更新学习率
        scheduler.step()


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha]) #适用于二分类
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha) #适用于多分类
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                          # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))     # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          path=None):
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
    plt.figure(dpi=320, figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize': 100, 'fontweight': "bold"})
    # 添加颜色条
    cbar = plt.colorbar()
    # 设置颜色条刻度字体大小
    cbar.ax.tick_params(labelsize=35)
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45, fontdict={'fontsize': 10})
    # plt.yticks(tick_marks, classes, rotation=45, fontdict={'fontsize': 10})
    plt.xticks(tick_marks, classes, rotation=45, fontsize=50, fontweight='bold')
    plt.yticks(tick_marks, classes, rotation=45, fontsize=50, fontweight='bold')
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else '.0f'
    # fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", verticalalignment="center",
                 color="#DE4F2E" if cm[i, j] > thresh else "#DE4F2E",
                 fontdict={'fontsize': 150})

    plt.tight_layout()
    plt.xlabel('Predicted label', fontdict={'fontsize': 70, 'fontweight': "bold"})
    plt.ylabel('True label', fontdict={'fontsize': 70, 'fontweight': "bold"})
    plt.subplots_adjust(left=0.14, right=0.92, bottom=0.25, top=0.85)
    # plt.show()
    plt.savefig(path, format="svg")