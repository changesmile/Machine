import numpy as np
import pandas as pd


def batch_gradient_descent(X, y, alpha, num_iters):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for i in range(num_iters):
        h = np.dot(X, theta)
        loss = h - y
        gradient = np.dot(X.T, loss) / m
        theta = theta - alpha * gradient
    return theta


pdData = pd.read_csv('./data/LogiReg_data.txt', header=None, names=['Exma1', 'Exma2', 'Admitted'])

pdData.insert(0, 'Ones', 1)
X = pdData.iloc[:, :3].values
Y = pdData.iloc[:, 3].values
n = 100
alpha = 0.000001
loss = batch_gradient_descent(X, Y, alpha, n)
print(loss)
