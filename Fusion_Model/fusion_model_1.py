"""
该文件用于初步的本地端leaf-ghostnet和云端模型的融合绘图工作
因为时间比较仓促的原因，所以暂且不用全连接层融合结果，先直接有对半分的权重来分析
"""
"""
此文件需要本地端生成一个配套的csv，然后从图片数据集中读取
"""
"""
现在是需要，因为一个音频文件，是可能有多张图片，目前是想算一下n和p概率的平均
"""
from 模型和网络 import ResNet, Covnet_3
from numpy import *
import csv
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import datetime
import itertools
import imageio.v2 as imageio


def count_mean_std(filepath):
    """
    :param filepath: 传入的单张图片的地址
    :return: 返回计算的方差和均值
    """
    R_channel = 0
    G_channel = 0
    B_channel = 0

    img = imageio.imread(filepath) / 255.0
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

    num = 224 * 224  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0

    img = imageio.imread(filepath) / 255.0
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)
    # print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    # print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
    mean = [R_mean, G_mean, B_mean]
    std = [R_var, G_var, B_var]
    return mean, std


df = pd.read_csv(r"C:\Users\ldt20\Desktop\data.csv")
# print(df)
print("一共有：{}个文件".format(len(df.index)))
print(df.at[0, 'uid'])
print(type(df.at[0, 'positive_rate']))

"""
整理成一个二维列表，里面是字典的形式
[[{"uid":xxxxx, "label": negative},{}],[]]
"""
uid_same = '123'
wav_dict = []
j = -1
for i in range(len(df.index)):
    uid = df.at[i, 'uid'][:-6]
    if uid != uid_same:
        uid_same = uid
        wav_dict.append({"uid": uid,
                         "positive_rate": [df.at[i, 'positive_rate']],
                         "label": df.at[i, 'label']})
    else:
        wav_dict[-1]["positive_rate"].append(df.at[i, 'positive_rate'])

print(wav_dict)




# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# -------------------------------------------------- #
# （0）参数设置
# -------------------------------------------------- #
# 权重参数路径
weights_path = r"C:\Users\ldt20\Desktop\训练权重保存\model_Track1+CoughVid logMel train_val最好的权重ResNet18网络.pth"
work_path = "D:/学习/大创/data/训练数据集/model/photo"

# 图片文件路径
img_dir_path = r"D:\学习\大创\data\训练数据集\data\Track1+CoughVid 谱图合集\测试集(over 4s)\Track1+CoughVid logMel test"
dir_path = os.path.basename(img_dir_path)
cd = os.path.exists(os.path.join(work_path, dir_path))
if cd:
    print("保存文件夹已存在")
else:
    print("创建保存文件夹")
    os.mkdir(os.path.join(work_path, dir_path))

# weights_path = "D:/学习/大创/data/训练数据集/model/pth/melspec(1000_50)/model_melspec(1000_50)最好的权重.pth"
# 预测索引对应的类别名称
# class_names = ['cat', 'dog']
class_names = ['negative', 'positive']

print("=" * 50)
print("=" * 50)
print("=" * 50)
"""
读取一下图片文件的地址
"""
image_len = len(wav_dict)
# img_path: 每一张图片文件的地址，现在设计为一个一维列表
img_path = []
for i in range(len(wav_dict)):
    # img_path.append([])
    label = wav_dict[i]['label']
    uid = wav_dict[i]['uid']
    img_path.append({"uid": uid,
                     "photo_path": [],
                     "label": label})

    label_path = os.path.join(img_dir_path, label)
    # 遍历文件夹的根目录、文件夹、文件
    for root, dirs, files in os.walk(label_path):
        for file in files:
            file_name = file[:-6]
            if file_name == uid:
                # 存入一部字典，包含两个键值对，分别是图片的绝对路径和对应的uid
                img_path[-1]["photo_path"].append(os.path.join(label_path, file))

print(img_path)
print("len_img_path:{}".format(len(img_path)))
print("len_wav_dict:{}".format(len(wav_dict)))

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# -------------------------------------------------- #
# （1）数据加载
# -------------------------------------------------- #
# 数据预处理
# 加载模型
model = ResNet.resnet18(num_classes=2, include_top=True)

# model = eca_ResNet.eca_resnet18(num_classes=2)
# 加载权重文件
model.load_state_dict(torch.load(weights_path, map_location=device))
# 模型切换成验证模式，dropout和bn切换形式
model.eval()
# 前向传播过程中不计算梯度
predict_clas = []
predict_scores = []
predict_names = []
pre_score = []
ture_labels = []
image = []
pre_score = []
acc = 0.0
# 用于存储每一张图片的positive得分，二维list
positive_scores = []
# 用于存储每一张图片的negative得分，二维list
negative_scores = []

# 初始化混淆矩阵
cnf_matrix = np.zeros([2, 2])

"""
遍历每一张图片，进行预测和结果融合
"""
for i in range(len(img_path)):
    positive_scores.append([])
    negative_scores.append([])

    for j in range(len(img_path[i]["photo_path"])):
        photo_path = img_path[i]['photo_path'][j]
        label = img_path[i]['label']
        frame = Image.open(photo_path)
        # img_mean, img_std = count_mean_std(photo_path)
        img_mean = [0.33478397, 0.55253255, 0.5409211]
        img_std = [0.36969644, 0.3965568, 0.33030042]
        data_transform = transforms.Compose([
            # 将输入图像的尺寸变成224*224
            transforms.Resize((224, 224)),
            # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
            transforms.ToTensor(),
            # 对每个通道的像素进行标准化，给出每个通道的均值和方差
            transforms.Normalize(mean=img_mean, std=img_std)])
        img = data_transform(frame)
        # 给图像增加batch维度 [c,h,w]==>[b,c,h,w]
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # 前向传播
            outputs = model(img)
            # 只有一张图就挤压掉batch维度
            outputs = torch.squeeze(outputs)
            # 计算图片属于2个类别的概率
            predict = torch.softmax(outputs, dim=0)

            # 准备绘制roc曲线所需要的数据
            positive_pre = outputs[1]
            positive_pre = positive_pre.detach().numpy()
            positive_pre = positive_pre.tolist()
            positive_scores[i].append(positive_pre)
            for k in range(len(wav_dict[i]["positive_rate"])):
                positive_scores[i].append(wav_dict[i]["positive_rate"][k])

            negative_pre = outputs[0]
            negative_pre = negative_pre.detach().numpy()
            negative_pre = negative_pre.tolist()
            negative_scores.append(negative_pre)


    # 计算positive的综合概率
    pre_score.append(mean(positive_scores[i]))
    if pre_score[i] >= 0.5:
        predict_cla = 1
    else:
        predict_cla = 0
    # 获取预测类别的名称
    predict_name = class_names[predict_cla]
    print("predict_name:{}".format(predict_name))

    if predict_name == 'negative':
        predict_y = 0
    else:
        predict_y = 1

    if wav_dict[i]["label"] == 'negative':
        label_ture = 0
        ture_labels += [0]
    else:
        label_ture = 1
        ture_labels += [1]

    if predict_name == wav_dict[i]["label"]:
        acc += 1

    cnf_matrix[predict_y][label_ture] += 1

print("预测分数有：{}个".format(len(pre_score)))
print(pre_score)

print("一共校验了" + str(image_len) + "张图片，其中正确的有" + str(acc) + "张")
acc = acc / image_len
print("acc:" + str(acc))

"""
绘制roc曲线
"""
clr_1 = 'tab:green'
clr_2 = 'tab:green'
clr_3 = 'k'
fpr, tpr, thersholds = roc_curve(ture_labels, pre_score)
roc_auc1 = auc(fpr, tpr)
roc_auc = roc_auc1 * 100
plt.plot(fpr, tpr, label=' (auc = {0:.2f})'.format(roc_auc), c=clr_1, alpha=1)
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
# plt.savefig("D:/学习/大创/data/训练数据集/model/photo/" + dir_path + "/model_ROC_" + str(nowTime) + ".jpg")
plt.show()

"""
绘制混淆矩阵
"""
Confusion_matrix_path = "D:/学习/大创/data/训练数据集/model/photo/" + dir_path + "/Confusion matrix" + str(
    nowTime) + ".jpg"


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
    plt.show()
    # plt.savefig(path)


# 第一种情况：显示百分比
# classes = ['cat', 'dog']
classes = ['negative', 'positive']
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')

# # 第二种情况：显示数字
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')
