import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fig = plt.figure()
# ax = fig.add_subplot(3, 3, 1)
# ax2 = fig.add_subplot(3, 3, 2)
# rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='r', alpha=0.3)
# circ = plt.Circle((0.7, 0.2),0.15, color='b', alpha=0.3)
# ax.add_patch(rect)
# ax2.add_patch(circ)
# plt.show()

# -----------------------------
# a = np.arange(10)
# print(a)
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.plot(a, a * 1.5, a, a * 2.5, a, a * 3.5, a, a * 4.5)
# plt.legend(['1.5x', '2.5x', '3.5x', '4.5x'])
# plt.show()
# -----------------------------

# x = np.linspace(-10, 10, 100)
# print(x)
# y = np.sin(x)
# print(y)
# plt.plot(x, y, marker="o")
# plt.show()
# -----------------------------
df = pd.read_csv('../datas/iris.csv', header=None)
X = df.iloc[:, [0, 2]].values
# print(X)

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='+', label='Virginica')
plt.xlabel('Sepal.Length')
plt.ylabel('Sepal.Width')
plt.legend(loc=2)
plt.show()
