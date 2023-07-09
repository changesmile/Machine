import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure()
# a = [2, 3, 6, 4, 1, 6, 4, 7, 8, 3, 9, 5, 4, 1, 1, 3, 6, 5, 7, 8, 8, 2, 3, 5, 7, 8, 1]
# colors = ['blue', 'red', 'yellow']
# for i in range(3):
#     start = i * 5
#     end = (i+1) * 5
#     plt.plot(range(5), a[start: end], c=colors[i], label=colors[i])
# plt.legend(loc='best')
# plt.show()

df = pd.read_csv('E:/pythonProject/机器学习/datas/iris.txt', header=None)
df.columns = ['LenSep', 'LenPet']
# print(df)
plt.rcParams['font.sans-serif'] = ['SimHei']
ax1 = fig.add_subplot(2, 2, 1)

plt.title("花瓣/花萼长度散点图")  # 图表标题
ax1.scatter(df['LenSep'], df['LenPet'], c='red')

ax2 = fig.add_subplot(2, 2, 2)
plt.title("花瓣长度直方图")
ax2.hist(df['LenSep'])


x = list(np.arange(30))
ax3 = fig.add_subplot(2, 2, 3)
plt.title("花萼长度条形图")
ax3.bar(x, height=df['LenPet'], width=0.5)

ax4 = fig.add_subplot(2, 2, 4)
plt.title("花瓣长度饼图")
labels = ['A', 'B', 'C', 'D', 'E', 'F']
ax4.pie(df['LenSep'][8:14], labels=labels)

plt.show()
