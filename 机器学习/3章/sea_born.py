import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sinplot():
    for i in range(1, 7):
        x = np.linspace(-10, 10, 100)
        # y = np.sin(x)
        plt.plot(x, np.sin(x + i * .5) * (7 - i))


# sns.set_style('dark')
# # 正态分布 20行6列
# data = np.random.normal(size=(20, 6))
# data2 = np.arange(6) / 2
# print(data+data2)
# sns.boxplot(data=data+data2)
# plt.show()
with sns.axes_style('darkgrid'):
    plt.subplot(2,1,1)
    sinplot()
plt.subplot(2,1,2)
sinplot()
plt.show()
