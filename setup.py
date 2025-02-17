from setuptools import setup, find_packages

setup(
    name='pris7',
    version='0.1',
    description='A simple MNIST CNN training and testing package',
    # long_description=open('README.md').read(),  # 可选，如果有 README 文件
    long_description_content_type='text/markdown',  # 如果使用 Markdown 格式
    author='Zhang Kailong',
    # author_email='your_email@example.com',
    # url='https://github.com/flzt11/pris7_MNIST',  # 如果有 GitHub 仓库链接
    packages=find_packages(),  # 自动发现所有包
    install_requires=[
        'torch==1.13.1+cu117',
        'torchvision==0.14.1+cu117',
        'torchaudio==0.13.1',
        'numpy',
        'opencv-python',
        'matplotlib',
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu117',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 根据你的环境选择合适的版本
)
