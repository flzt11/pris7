# pris7
 AI课程任务7-基于Mnist的手写数字识别

使用说明
conda create -n pris7 python=3.7
conda activate pris7
git clone https://github.com/flzt11/pris7.git
pip install ./pris7 --extra-index-url https://download.pytorch.org/whl/cu117

启动说明
import pris7
pris7.cv("Train Mnist Dataset")  # Start training
pris7.cv("Test Mnist Dataset")   # Start testing
