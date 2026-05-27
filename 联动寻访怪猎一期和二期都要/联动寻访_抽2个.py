import numpy as np

from random_variable import FiniteDist

# 161 表示怪猎一期，只要 6 星，只要 1 个，其他同理
d161 = FiniteDist(pk=np.load("161.npy"))
d1a1 = FiniteDist(pk=np.load("1a1.npy"))
d1a6 = FiniteDist(pk=np.load("1a6.npy"))
d261 = FiniteDist(pk=np.load("261.npy"))
d2a1 = FiniteDist(pk=np.load("2a1.npy"))
d2a6 = FiniteDist(pk=np.load("2a6.npy"))

# 用卷积计算随机变量的和的分布
da61 = d161 * d261
daa1 = d1a1 * d2a1
daa6 = d1a6 * d2a6

for d in [da61, daa1, daa6]:
    print(d.cdf(np.arange(0, 2048)).round(4).tolist())
