import torch
import torchvision
import numpy as np
import cv2
from pris7.model import CNN
import os

def test_mnist():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    test_data_dir = os.path.join(parent_dir, 'pris7', 'data')
    test_data = torchvision.datasets.MNIST(
        root=test_data_dir,
        train=False  # 表明是测试集111
    )

    test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
    # torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
    # 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
    # test_y = test_data.test_labels[:2000]

    # 加载训练好的模型
    cnn = CNN()

    # 获取当前文件的目录


    # 构造相对路径
    model_path = os.path.join(parent_dir, 'pris7', 'cnn_model.pth')

    # 加载模型
    cnn.load_state_dict(torch.load(model_path))

    cnn.eval()

    inputs = test_x[:32]  # 测试32个数据
    # 加载并预测
    test_output = cnn(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')

    # 将图像从tensor转换为numpy数组
    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)

    # 假设你有32张图片，每行显示8个数字
    images_per_row = 8
    img_height, img_width = 25, 25  # MNIST 图像尺寸为 28x28

    # 将图片转换为 8 位无符号整数类型 (0-255)
    img = (img * 255).astype(np.uint8)

    # 如果图像是RGB，确保其格式正确
    if img.shape[2] == 3:  # 确保是RGB图像
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 显示图像并添加预测数字文本
    for i in range(32):  # 假设一次测试32张图片
        # 获取每张图片的预测结果
        prediction = str(pred_y[i])

        # 计算当前数字应显示的行和列
        row = i // images_per_row  # 当前行数
        col = i % images_per_row  # 当前列数

        # 计算文本的大小
        text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_width, text_height = text_size

        # 计算文本的中心坐标
        text_x = col * (img_width + 5) + (img_width - text_width) // 2  # 水平居中
        text_y = row * (img_height + 5) + (img_height + text_height) // 2  # 垂直居中

        # 添加文本
        cv2.putText(img, prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # 显示结果
    cv2.imshow('Predictions', img)  # opencv显示需要识别的数据图片
    key_pressed = cv2.waitKey(0)
    cv2.destroyAllWindows()
