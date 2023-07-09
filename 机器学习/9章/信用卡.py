# -*- encoding:utf-8 -*-
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def loaddata():
    people = pd.read_csv('E:/pythonProject/机器学习/datas/credit-overdue.csv')  # 加载数据集
    X = people[['debt', 'income']].values
    y = people['overdue'].values
    return X, y


print("Step1:read data...")
x, y = loaddata()

# 拆分为训练数据和测试数据
print("Step2:fit by Perceptron...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 将两类值分别存放、以便显示
positive_x1 = [x[i, 0] for i in range(len(y)) if y[i] == 1]
positive_x2 = [x[i, 1] for i in range(len(y)) if y[i] == 1]
negetive_x1 = [x[i, 0] for i in range(len(y)) if y[i] == 0]
negetive_x2 = [x[i, 1] for i in range(len(y)) if y[i] == 0]

# 定义感知机
clf = Perceptron()
clf.fit(x_train, y_train)
print("Step3:get the weights and bias...")

# 得到结果参数
weights = clf.coef_
bias = clf.intercept_
print('  权重为：', weights, '\n  截距为：', bias)
print("Step4:compute the accuracy...")

# 使用测试集对模型进行验证
acc = clf.score(x_test, y_test)
print('  精确度：%.2f' % (acc * 100.0))

# 绘制两类样本散点图
print("Step5:draw with the weights and bias...")
plt.scatter(positive_x1, positive_x2, marker='^', c='red')
plt.scatter(negetive_x1, negetive_x2, c='blue')

# 显示感知机生成的分类线
line_x = np.arange(0, 4)
line_y = line_x * (-weights[0][0] / weights[0][1]) - bias
plt.plot(line_x, line_y)
plt.show()