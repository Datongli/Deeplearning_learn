import torch
import torch.nn as nn
from torch.nn import init
from torchstat import stat  # 查看网络参数
from torchsummary import summary  # 查看网络结构

'''
    建立模型
    (batch_size,input_channels,step)
    两秒的音频，step=2*16000=32000
'''


class TCNN4(nn.Module):
    def __init__(self):
        super(TCNN4, self).__init__()  # 继承父类的初始化方法
        # b*1*32000 -> b*32*10666
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=3, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        # b*32*10666 -> b*32*5330
        self.pool1 = nn.MaxPool1d(kernel_size=7, stride=2)
        # b*32*5330 -> b*64*5312
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        # b*64*5312 -> b*64*5306
        self.pool2 = nn.MaxPool1d(kernel_size=7, stride=1)
        # b*64*5306 -> b*128*2634
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=2, padding=1, dilation=5)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        # b*128*2634 -> b*128*2628
        self.pool3 = nn.AvgPool1d(kernel_size=7, stride=1)
        # b*128*2628-> b*128*2614
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, stride=1, padding=1, dilation=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        # b*128*2614 -> b*128*2608
        self.pool4 = nn.MaxPool1d(kernel_size=7, stride=1)
        # b*128*2608 -> b*128*1293
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, stride=2, padding=1, dilation=3)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        # b*128*1293 -> b*128*1287
        self.pool5 = nn.MaxPool1d(kernel_size=7, stride=1)
        # b*128*1387 -> b*128*1273
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, stride=1, padding=1, dilation=2)
        self.bn6 = nn.BatchNorm1d(128)
        self.relu6 = nn.ReLU()
        # b*128*1273 -> b*128*634
        self.pool6 = nn.MaxPool1d(kernel_size=7, stride=2)
        # b*128*634 -> b*128*306
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, stride=2, padding=1, dilation=3)
        self.bn7 = nn.BatchNorm1d(128)
        self.relu7 = nn.ReLU()
        # b*128*306 -> b*128*150
        self.pool7 = nn.MaxPool1d(kernel_size=7, stride=2)
        # b*128*150 -> b*128*136
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, stride=1, padding=1, dilation=2)
        self.bn8 = nn.BatchNorm1d(128)
        self.relu8 = nn.ReLU()
        # b*128*136 -> b*128*65
        self.pool8 = nn.MaxPool1d(kernel_size=7, stride=2)
        # b*128*65 -> b*128*26
        self.conv9 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, stride=2, padding=1, dilation=2)
        self.bn9 = nn.BatchNorm1d(128)
        self.relu9 = nn.ReLU()
        # b*128*26 -> b*128*10
        self.pool9 = nn.MaxPool1d(kernel_size=7, stride=2)
        # b*128*10 -> b*128*8
        self.conv10 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=2)
        self.bn10 = nn.BatchNorm1d(128)
        self.relu10 = nn.ReLU()
        # b*128*8 -> b*128*2
        self.pool10 = nn.MaxPool1d(kernel_size=5, stride=2)
        #
        self.fc1 = nn.Linear(128 * 2, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        # 权重随机初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        inputs_size = x.size(0)
        x = self.conv1(x)
        # x = self.dropout(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.pool9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.pool10(x)
        x = x.view(-1, 128 * 2)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 接收网络模型
    model = TCNN4()
    # print(model)
    # 查看网络参数量，不需要指定输入特征图像的batch维度
    # stat(model, input_size=(1, 1, 32000))

    # 查看网络结构及参数
    summary(model, input_size=[(1, 1, 32000)], device='cpu')
