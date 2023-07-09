import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn import linear_model
from numpy import *

# def DrawPlt():
#     plt.figure()
#     plt.title('Cost and Income Of a Film')
#     plt.xlabel('Cost(Million Yuan)')
#     plt.ylabel('Income(Million Yuan)')
#     # x轴：0-25， y轴：0-60
#     plt.axis([0, 25, 0, 60])
#     # 显示网格
#     plt.grid(True)
#
#
# X = [[6], [9], [12], [14], [16]]
# y = [[9], [12], [29], [35], [59]]
#
# model = linear_model.LinearRegression()
# model.fit(X, y)
# predict_cost = model.predict([[20]])
# # print(predict_cost[0][0])
# w = model.coef_[0][0]
# b = model.intercept_[0]
# print("投资2千万的电影预计票房收入为：{:.2f}百万元".format(predict_cost[0][0]))
# print("回归模型的系数是：", w)
# print("回归模型的截距是：", b)
# DrawPlt()
# # '.'以点的形式表现，而不是默认的线
# plt.plot(X, y, '.')
#
# plt.plot([0, 25], [b, 25 * w + b])
# plt.show()

# import numpy as np
# from sklearn import datasets, linear_model
#
# x = np.array([[6, 1, 9], [9, 3, 12], [12, 2, 29],
#               [14, 3, 35], [16, 4, 59]])
# X = x[:, :2]  # 自变量，多个X ,称为多元
# Y = x[:, -1]  # 因变量
# print('X:', X)
# print('Y:', Y)
# model = linear_model.LinearRegression()
# model.fit(X, Y)
# print('系数(w1,w2)为:', model.coef_)
# print('截距(b)为:', model.intercept_)
# # 预测
# y_predict = model.predict(np.array([[10, 3]]))
# print('投资1千万，推广3百万的电影票房预测为：', y_predict, '百万')
# plt.plot([0, 2], [2, 4])
# plt.show()
def func(line_x, regr):
    lore = regr.coef_
    y1 = 0
    for w in line_x:
        y1 = y1 + w[0]*lore[0] + w[1]*lore[1] + w[2]*lore[2]
    y1 = y1 + regr.intercept_
    y2 = 1 / (1 + exp(-y1))
    return y2



data = np.array([
    [20, 7000, 800, 1],
    [35, 200, 2500, 0],
    [27, 5000, 3000, 1],
    [32, 4000, 4000, 0],
    [45, 2000, 3800, 0],

])
test_data = [30, 3500, 3500, ]

x = data[:, :3]
y = data[:, 3]
# print(x)
# print(y)
regr = linear_model.LinearRegression()
regr.fit(x,y)
# print('系数', regr.coef_)
# print('截距', regr.intercept_)
line_y = func(x, regr)
plt.plot([-10, 10], [0, line_y])
plt.show()