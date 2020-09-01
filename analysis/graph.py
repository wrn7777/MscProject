import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return np.minimum(np.maximum(x, 0), 6)


x=np.arange(-10, 16, 0.01)
y=function(x)
plt.plot(x,y)


#plt.xlim&ylim可以限制x轴或者y轴的范围，相当于把图像从某个小区域进行放大，或者从某个更大的区域缩小
# plt.xlim(-5,5)
# plt.ylim(0,100)
plt.show()