import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    data = 1 / (1 + np.exp(-z))
    return data


def model(X, theta):
    data = sigmoid(np.dot(X, theta.T))
    return data


pdData = pd.read_csv('./data/LogiReg_data.txt', header=None, names=['Exma1', 'Exma2', 'Admitted'])

pdData.insert(0, 'Ones', 1)


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    data = model(X, theta)
    error = (data - y)
    for j in range(len(theta.ravel())):  # for each parmeter
        term = np.multiply(error, X[:, j])
        grad[0, j] = np.sum(term) / len(X)

    return grad


X = pdData.iloc[:, :3].values
Y = pdData.iloc[:, 3].values
theta = np.zeros([1, 3])

import time
import numpy.random

# 洗牌
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stopCriterion(type, value, threshold):
    # 设定三种不同的停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


def shuffleData(data):
    X = data.iloc[:, :3].values
    y = data.iloc[:, 3].values
    return X, y


def descent(data, theta, batchSize, stopType, thresh, alpha):
    # 梯度下降求解

    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值

    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
        k += batchSize  # 取batch数量个数据
        if k >= n:
            k = 0
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1

        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh): break

    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)

    return costs


n = 100
costs = runExpe(pdData, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
print(costs)

#设定阈值
def predict(X, theta):
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]