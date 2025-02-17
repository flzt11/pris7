import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义第一个卷积层 -> 激活函数 -> 池化层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),  # 输入为1通道，输出为16通道，卷积核为5x5，步长1，padding为2
            nn.ReLU(),  # 激活函数ReLU
            nn.MaxPool2d(2)  # 池化层，2x2的池化
        )

        # 定义第二个卷积层 -> 激活函数 -> 池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 输入为16通道，输出为32通道
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 定义全连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输入为32 * 7 * 7，输出为10个类

    def forward(self, x):
        # 定义前向传播过程
        x = self.conv1(x)  # 通过第一个卷积层
        x = self.conv2(x)  # 通过第二个卷积层
        x = x.view(x.size(0), -1)  # 展平多维的输入为一维
        output = self.out(x)  # 通过全连接层
        return output
