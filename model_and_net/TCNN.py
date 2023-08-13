import torch
import torch.nn as nn
from torchstat import stat  # 查看网络参数
from torchsummary import summary  # 查看网络结构
import torch.nn.functional as F

'''
    建立模型
    (batch_size,input_channels,step)
    两秒的音频，step=2*16000=32000
'''


class TCNN(nn.Module):
    def __init__(self, out_num=2, dropout=0.2):
        super(TCNN, self).__init__()  # 继承父类的初始化方法
        self.dropout_num = dropout
        # b*1*32000 -> b*32*32000
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # b*32*32000 -> b*32*32000
        self.bn1 = nn.BatchNorm1d(32)
        # b*32*32000 -> b*32*32000
        self.relu1 = nn.ReLU()
        # b*32*32000 -> b*32*10666
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        # b*32*10666 -> b*64*10666
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # b*64*10666 -> b*64*10666
        self.bn2 = nn.BatchNorm1d(64)
        # b*64*10666 -> b*64*10666
        self.relu2 = nn.ReLU()
        # b*64*10666 -> b*64*2666
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        # b*64*2666 -> b*64*2666*1
        # self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 188), stride=188)

        self.fc1 = nn.Linear(64 * 2666, 128)
        self.relu3 = nn.ReLU()
        # fc2为全连接层，128为输入通道数，2为输出通道数
        self.fc2 = nn.Linear(128, out_num)
        # self.dropout = nn.Dropout(self.dropout_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        inputs_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.dropout(x)
        x = self.pool2(x)

        # Dumped
        # x = x.unsqueeze(0).unsqueeze(0)
        # x = self.conv3(x)
        # return x

        x = x.view(-1, 64 * 2666)
        x = self.fc1(x)
        x = self.relu3(x)
        if self.dropout_num > 0.:
            x = F.dropout(x, p=self.dropout_num, training=self.training)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x


