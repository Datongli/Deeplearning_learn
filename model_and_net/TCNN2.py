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


class TCNN2(nn.Module):
    def __init__(self, out=2, dropout=0.2):
        super(TCNN2, self).__init__()  # 继承父类的初始化方法
        self.dropout_num = dropout
        # b*1*32000 -> b*32*10615
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=3, padding=2, dilation=20)
        # b*32*10615 -> b*32*10615
        self.bn1 = nn.BatchNorm1d(32)
        # b*32*10615 -> b*32*10615
        self.relu1 = nn.ReLU()
        # b*32*10615 -> b*32*5037
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        # b*32*5037 -> b*64*5037
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3)
        # b*64*5037 -> b*64*5037
        self.bn2 = nn.BatchNorm1d(64)
        # b*64*5037 -> b*64*5037
        self.relu2 = nn.ReLU()
        # b*64*5037 -> b*64*5304
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=1)
        # b*64*5304 -> b*128*2648
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=5)
        # b*128*2648 -> b*128*2648
        self.bn3 = nn.BatchNorm1d(128)
        # b*128*2648 -> b*128*2648
        self.relu3 = nn.ReLU()
        # b*128*2648 -> b*128*2646
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128 * 2646, 128)
        self.dropout = nn.Dropout(0.3)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, out)

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
        x = x.view(-1, 128 * 2646)
        if self.dropout_num > 0.:
            x = F.dropout(x, p=self.dropout_num, training=self.training)
        x = self.fc1(x)
        x = self.relu3(x)
        if self.dropout_num > 0.:
            x = F.dropout(x, p=self.dropout_num, training=self.training)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    model = TCNN2()
    x = torch.randn(5, 1, 32000)
    y = model(x)
    print(y)
