from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(z):
    data = 1 / (1 + np.exp(-z))
    return data


def model(X, W):
    return np.dot(X, W)


def loss(h, y):
    data = -y * np.log(h) - (1 - y) * np.log(1 - h)
    return data.mean()


def gradient(x, h, y):
    data = np.dot(x.T, (h - y)) / y.shape[0]
    return data


def Logistic_Regression(x, y, lr, num_iter):
    n, m = x.shape
    intercept = np.ones((n, 1))
    x = np.concatenate((intercept, x), axis=1)
    w = np.zeros(m + 1)
    l =0
    for i in range(num_iter):
        h = sigmoid(model(x, w))
        g = gradient(x, h, y)
        l = loss(h, y)
        w = w - lr * g

    return l, w


df = pd.read_csv('./data/credit-overdue.csv')
x = df[['debt', 'income']].values
y = df['overdue'].values
lr = 0.001
num_iter = 10000
Logistic_Regression(x, y, lr, num_iter)
