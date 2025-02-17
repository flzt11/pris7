# pris7
AI课程任务7 - 基于Mnist的手写数字识别

## 使用说明
1. 创建并激活虚拟环境：
    ```bash
    conda create -n pris7 python=3.7
    conda activate pris7
    ```

2. 克隆项目并安装依赖：
    ```bash
    git clone https://github.com/flzt11/pris7.git
    pip install ./pris7 --extra-index-url https://download.pytorch.org/whl/cu117
    ```

## 启动说明
1. 导入并启动训练和测试：
    ```python
    import pris7
    pris7.cv("Train Mnist Dataset")  # 启动训练
    pris7.cv("Test Mnist Dataset")   # 启动测试
    ```

## 依赖项
- Python 3.7
- Conda
- PyTorch (CUDA 11.7)
