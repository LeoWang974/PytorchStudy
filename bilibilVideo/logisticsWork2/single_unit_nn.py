import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py
from PIL import Image
from scipy import ndimage
from data import load_dataset

"""
实现一个学习算法的整体结构
1.获取并定义模型输入
2.初始化参数
3.计算成本函数及其梯度
4.使用优化算法（梯度下降）
    4.1循环
    4.2计算当前损失（正向传播）
    4.3计算当前梯度（反向传播）
    4.4更新参数（梯度下降）

读取数据要求
1.按照向量化伪代码实现的形状要求将样本数据进行转换
2.标准化数据处理
"""

train_x, train_y, test_x, test_y, classes = load_dataset()
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
