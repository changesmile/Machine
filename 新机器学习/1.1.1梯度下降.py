import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(X, y, theta):
    sig = model(X, theta)
    cost_data = (np.multiply(-y, np.log(sig)) - np.multiply(1 - y, np.log(1 - sig))).mean()
    return cost_data


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


def gradient(X, h, y):
    gradient_data = np.dot()


pdData = pd.read_csv('./data/LogiReg_data.txt', header=None, names=['Exma1', 'Exma2', 'Admitted'])

# positive = pdData[pdData['Admitted'] == 1]
# negative = pdData[pdData['Admitted'] == 0]
# fig = plt.figure(figsize=(10, 5))
# plt.scatter(positive['Exma1'], positive['Exma2'], s=30, marker='o', c='green', label='Admitted')
# plt.scatter(negative['Exma1'], negative['Exma2'], s=30, marker='x', c='red', label='Not Admitted')
# plt.legend(['Admitted', 'Not Admitted'])
# plt.xlabel('Exma1 Score')
# plt.ylabel('Exma2 Score')
# positive = pdData[pdData['Admitted'] == 1]
# negative = pdData[pdData['Admitted'] == 0]
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.scatter(positive['Exma1'], positive['Exma2'], s=30, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exma1'], negative['Exma2'], s=30, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

pdData.insert(0, 'Ones', 1)
X = pdData.iloc[:, :3].values
Y = pdData.iloc[:, 3].values
theta = np.zeros([1, 3])
print(cost(X, Y, theta))
