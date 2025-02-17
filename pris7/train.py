import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os
from pris7.model import CNN

def train_mnist():
    torch.manual_seed(1)

    # 超参数
    EPOCH = 1
    BATCH_SIZE = 50
    LR = 0.001
    DOWNLOAD_MNIST = True

    # 下载mnist手写数据集
    train_data = torchvision.datasets.MNIST(
        root='./data/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
    )

    test_data = torchvision.datasets.MNIST(root='./data/', train=False)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')

    torch.save(cnn.state_dict(), 'cnn_model.pth')
