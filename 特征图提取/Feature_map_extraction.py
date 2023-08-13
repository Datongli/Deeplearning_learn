"""
此文件用于模型中特征图的提取
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import ResNet
import datetime
import os



# 权重路径
weightpath = r'D:\学习\大创\data\训练数据集\model\pth\logMel 1：1\model_logMel 1：1_第1折验证ResNet网络.pth'

model = ResNet.resnet18(num_classes=2, include_top=True)

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')
# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


# 加载权重
model.load_state_dict(torch.load(weightpath, map_location=device))

print(model)

# 从测试集中读取一张图片，并显示出来
img_path = r'D:\学习\大创\data\训练数据集\data\NeurIPS2021 谱图合集\logMel\negative\cough_00CwMwLaxlh6_225021680817-16.0K-VAD-1.jpg'
img = Image.open(img_path)
save_path = r"C:\Users\ldt20\Desktop\test"
imgarray = np.array(img) / 255.0
plt.figure(figsize=(8,8))
plt.imshow(imgarray)
plt.axis('off')
plt.savefig(os.path.join(save_path, "原图片"+nowTime))
plt.show()

# 将图片处理成模型可以预测的形式
# 图片处理
transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize([0.42746902, 0.6474644, 0.4993739], [0.38065454, 0.35554656, 0.3535182])
])

input_img = transform(img).unsqueeze(0)
print("输入图片的大小")
print(input_img.shape)

# 定义钩子函数，获取指定层名称的特征
activation = {}  # 保存获取的输出
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model.eval()
# 获取layer1里面的bn2层的结果，浅层特征
model.layer1[1].register_forward_hook(get_activation('bn2')) # 为layer1中第2个模块的bn3注册钩子
_ = model(input_img)
bn2 = activation['bn2'] # 结果将保存在activation字典中
print("所提取的层的大小")
print(bn2.shape)
# 可视化结果，显示前16张
plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(bn2[0, i, :, :], cmap='gray')
    plt.axis('off')
plt.show()
# 很多图片都可以分辨出原始图片包含的内容，这说明浅层网络能够提取图像较大粒度的特征


# 获取深层的特征映射
model.eval()
# 获取layer4中第3个模块的bn3层输出结果
model.layer4[1].register_forward_hook(get_activation('bn2'))
_ = model(input_img)
bn2 = activation['bn2']
# 绘制前64个特征
plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(bn2[0, i, :, :], cmap='gray')
    plt.axis('off')
plt.show()
# 可以发现深层的映射已经不能分辨图像的具体内容，说明更深层的特征映射能从图像中提取更细粒度的特征


"""
下面是计算模型的热力图
"""
class GradCAM(nn.Module):
    def __init__(self):
        super(GradCAM, self).__init__()
        # 获取模型的特征提取层
        """
        这段代码是用来创建一个神经网络模型的序列，
        其中包含了原始模型中除了全局平均池化层（'avgpool'）、全连接层（'fc'）和softmax层以外的所有层。
        具体来说，代码使用了Python中的OrderedDict函数来创建一个序列对象，
        其中每一个元素都是一个由模型中的子模块名称和对应的子模块层组成的字典。
        这里使用了模型的named_children()函数来获取模型的所有子模块，然后通过if语句来筛选掉需要剔除的层，
        最终将剩余的层按照原有顺序加入到序列中。
        """
        self.feature = nn.Sequential(OrderedDict({
            name: layer for name, layer in model.named_children()
            if name not in ['avgpool', 'fc', 'softmax']
        }))
        # 获取模型最后的平均池化层
        self.avgpool = model.avgpool
        # 获取模型的输出层
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', model.fc),
            ('softmax', model.softmax)
        ]))
        # 生成梯度占位符
        self.gradients = None

    # 获取梯度的钩子函数
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.feature(x)
        # 注册钩子
        h = x.register_hook(self.activations_hook)
        # 对卷积后的输出使用平均池化
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # 获取梯度的方法
    def get_activations_gradient(self):
        return self.gradients

    # 获取卷积层输出的方法
    def get_activations(self, x):
        return self.feature(x)


# 获取热力图
def get_heatmap(model, img):
    model.eval()
    img_pre = model(img)
    # 获取预测最高的类别
    pre_class = torch.argmax(img_pre, dim=-1).item()
    # 获取相对于模型参数的输出梯度
    img_pre[:, pre_class].backward()
    # 获取模型的梯度
    gradients = model.get_activations_gradient()
    # 计算梯度相应通道的均值
    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # 获取图像在相应卷积层输出的卷积特征
    activations = model.get_activations(input_img).detach()
    # 每个通道乘以相应的梯度均值
    for i in range(len(mean_gradients)):
        activations[:, i, :, :] *= mean_gradients[i]
    # 计算所有通道的均值输出得到热力图
    heatmap = torch.mean(activations, dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

cam = GradCAM()
# 获取热力图
heatmap = get_heatmap(cam, input_img)

# 可视化热力图
plt.matshow(heatmap)
plt.savefig(os.path.join(save_path, "热力图"+nowTime))
plt.show()


# 合并热力图和原题，并显示结果
def merge_heatmap_image(heatmap, image_path):
    # img = cv2.imread(image_path)
    img = Image.open(image_path)
    print(img)
    print(type(img))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    grad_cam_img = heatmap * 0.4 + img
    grad_cam_img = grad_cam_img / grad_cam_img.max()
    # 可视化图像
    b,g,r = cv2.split(grad_cam_img)
    grad_cam_img = cv2.merge([r,g,b])

    plt.figure(figsize=(8,8))
    plt.imshow(grad_cam_img)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, "原图与热力图组合"+nowTime))
    plt.show()


merge_heatmap_image(heatmap, img_path)
print("over")



# # 加载图像
# img = cv2.imread(r'D:\学习\大创\data\训练数据集\data\logMel 1：1\negative\aADACozj_cough-16.0K-VAD-0.jpg')
#
# # 预处理图像
# transform = transforms.Compose([
#     transforms.Resize([224,224]),
#     transforms.ToTensor(),
#     transforms.Normalize([0.42746902, 0.6474644, 0.4993739], [0.38065454, 0.35554656, 0.3535182])
# ])
# img_tensor = transform(img).unsqueeze(0)
#
# # 获取特征图
# model.eval()
# with torch.no_grad():
#     feature_map = model.features(img_tensor)
#
# # 将特征图转换为numpy数组
# feature_map = feature_map.squeeze(0).numpy()
#
# # 对特征图进行归一化处理
# feature_map -= np.min(feature_map)
# feature_map /= np.max(feature_map)
#
# # 将特征图上采样至原图像大小
# feature_map = cv2.resize(feature_map, (img.shape[1], img.shape[0]))
#
# # 将特征图映射到RGB图像上
# heatmap = cv2.applyColorMap(np.uint8(255*feature_map), cv2.COLORMAP_JET)
#
# # 将特征图与RGB图像融合
# result = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
#
# # 显示结果图像
# plt.imshow(result)
# plt.axis('off')
# plt.show()

