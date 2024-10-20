import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py
from PIL import Image
from mpmath import sigmoid
from scipy import ndimage
from datasets import load_dataset

"""
逻辑回归
1.逻辑回归函数
    1.1输入特征变量 x ∈Rnx
    1.2参数值：w∈Rnx，偏置值：b∈R
    1.3输出y^（y尖）=σ（WT(转置)x + b） = σ(w1x1+w2x2+……+wnxn+b) = σ(θTx)
2.sigmoid函数,通过此函数可以的得到预测结果
    s = σ（WT(转置)x + b） = σ(z) = 1/(1+e^(-z)) = J（w,b）  
3.损失函数L（y^,y） = -(ylogy^) - (1-y)log(1-y^)
    3.1若y=1，L（y^,y） = -logy^，若想L越小，y^必须越大，与趋近于1
    3.2若y=0，L（y^,y） = log(1-y^)，若想L越小，y^必须越小，与趋近于0
4.代价函数（cost）/成本函数
    J（w,b） = 1/m{∑(i=1 ~ m) L（y^,y）}    也就是总的损失 
5.梯度下降
    目的：使损失函数的值找到最小值 凸函数
    训练模型的目的是寻找合适的w与b以最小化代价函数值
    w = w - α{dJ(w,b)/dw}, b = b - {dJ(w,b)/db}，根据这个进行梯度更新
    α为学习速率，即每次更新的w步长。当w大于最优解w*时，导数大于0，那么w就会向更小的方向更新，反之向更大的方向更新，通过导数进行梯度下降，直至导数为0
    第二行中的逻辑回归梯度下降公式是由代价函数J对Sigmund函数进行求导得出的
6.逻辑回归的梯度下降计算过程
    6.1使用公式：
    z = WTx + b          即输入，参数矩阵乘输入特征加偏置值
    y^ = a = σ(z) = sigmoid(z)   即预测结果,此处使用sigmoid函数作为σ
    J(a,y) = L(a,y) = L（y^, y） = -(yloga) - (1-y)log(1-a)        见上式，a实际上就是y^，用于替代
    对于逻辑回归，先由W,x,b计算得到z，再由sigmoid函数对z进行处理得到y^即a，最后再得到损失函数L(y^, y)=L(a, y)=J(a,y)
    6.2参数的导数：
    从后向前求导，dJ/dz = (dJ/da)/(da/dz) = a-y,其中dJ/da = -y/a + (1-y)/(1-a), da/dz = a(1-a)
    由此可以求出总损失J相对于w1，w2……,b等参数的某一点导数，即
    dJ/dw = dJ/dz * dz/dw = dz*x        dJ/db = dz
    再跟据上式进行梯度更新 w = w - α{dJ(w,b)/dw}, b = b - {dJ(w,b)/db}
"""

"""
7.向量化编程，使用numpy
    对于m个样本，进行梯度下降时对每个样本分别计算一次，得到m个样本的w1，w2……和b的梯度，然后对求期望值进行梯度下降
"""
a = np.random.rand(100000)
b = np.random.rand(100000)

#法一，for循环
c = 0
start = time.time()
for i in range(len(a)):
    c += a[i] * b[i]
end = time.time()
print("计算所用时间%s "  %str(1000*(end-start)) + 'ms')

#法二，使用向量化方法
start = time.time()
c = np.dot(a,b)
end = time.time()
print("计算所用时间%s "  %str(1000*(end-start)) + 'ms')


"""
8.正向传播与反向传播
    正向传播：从前往后计算出梯度与损失
    反向传播：从后往前计算参数的更新梯度值
"""

"""
作业
"""

#使用np.exp()实现sigmoid函数
def basic_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

#实现sigmoid gradient(梯度)----即sigmoid的导数
def sigmoid_derivative(x):
    s = basic_sigmoid(x)
    ds = s * (1 - s)
    return ds

sigmoidtemp = basic_sigmoid(np.array([1, 2, 3]))
sigmoidgra = sigmoid_derivative(np.array([1, 2, 3]))
print(sigmoidtemp)
print(sigmoidgra)