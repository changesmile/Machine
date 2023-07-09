import inline as inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('E:/pythonProject/机器学习/datas/seaborn-data-master/tips.csv')  # 加载数据集

# sns.set(rc={"figure.figsize": (6, 6)})
# hls 用来调色 是个默认的颜色空间   ， 9 为显示的个数
# current_color2 = sns.color_palette('hls', 6)
# current_color = sns.hls_palette(6, l=.7, s=.9)
# # sns.palplot(current_color)
# data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
# sns.boxplot(data=data, palette=current_color)
# plt.show()


# ------------------

# # 由浅到深
# current_color = sns.color_palette('Reds')
# # 由深到浅
# current_color = sns.color_palette('Reds_r')
# sns.palplot(current_color)
# plt.show()

# ------------------

# x = np.random.normal(size=100)
# # bins 分成几等分
# sns.displot(x, bins=20, kde=False)
# plt.show()


# ------------------
# mean ,cov = [0, 1], ([1, .5], [.5, 1])
# # 生产一个正态矩阵分布 ， 200个，
# data = np.random.multivariate_normal(mean, cov, 50)
# df = pd.DataFrame(data=data, columns=['x', 'y'])
# sns.jointplot(x='x', y='y', data=df)
# plt.show()

# ------------------
# mean ,cov = [0, 1], ([1, .5], [.5, 1])
# # 生产一个正态矩阵分布 ， 200个，
# x, y = np.random.multivariate_normal(mean, cov, 1000).T
# # print(x,y)
# #  kind='hex' 蜂窝形状，散点图数据量太大看不出来，所以用蜂窝形状
# with sns.axes_style('ticks'):
#     sns.jointplot(x=x, y=y, kind='hex', color='r')
# plt.show()
# ------------------

# data = pd.read_csv('E:/pythonProject/机器学习/datas/iris.csv')  # 加载数据集
# sns.pairplot(data)
# plt.show()
# ------------------

# 回归函数
# sns.regplot(x=data['total_bill'], y=data['tip'])
# hue 是用来控制第三种变量颜色
# sns.stripplot(x=data['day'], y=data['total_bill'], hue='sex', data=data)
# sns.catplot(x=data['day'], y=data['total_bill'], data=data, kind='strip')
# sns.catplot(x=data['day'], y=data['total_bill'], data=data, kind='swarm')
# sns.stripplot(x='day', y='total_bill', data=data)
# plt.show()

# ------------------
# 绘制 以day为分类的图，col_warp每行两个
# g = sns.FacetGrid(data, col="day", col_wrap=1, hue='time')
# g.map(sns.regplot, "total_bill", "tip")
# g.add_legend()
# plt.show()

# 申请可以使用中文
# sns.set(rc={"font.sans-serif": "simhei"})
#
# g3 = sns.FacetGrid(data, col="day", hue="time")
# g3.map(plt.scatter, "total_bill", "tip")
# new_labels = ['午餐', '晚餐']
# g3.add_legend(title="时间")
# for t, l in zip(g3._legend.texts, new_labels):
#     t.set_text(l)
# ------------------
# from pandas import Categorical
# ordered_days = data.day.value_counts().index
# print (ordered_days)
# ordered_days = Categorical(['Thur', 'Fri', 'Sat', 'Sun'])
# g = sns.FacetGrid(data, row="day", aspect=4,)
# g.map(sns.boxplot, "total_bill", order=ordered_days)
# plt.show()


#-------------------

g = sns.FacetGrid(data, hue="time", height=10, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend()
plt.show()