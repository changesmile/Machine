import numpy as np
import matplotlib.pyplot as plt

data = np.zeros((3, 4))
data2 = np.array([[1, 2], [3, 4]])
data3 = np.array([[5, 6], [7, 8]])
data4 = np.hstack((data2, data3))  # 横着拼
data5 = np.vstack((data2, data3))  # shu着拼

data6 = data.copy()
# print(id(data))
# print(id(data6))

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(np.random.randint(1, 5, 5), range(5), c='blue')
ax2.plot(np.random.randint(1, 5, 5), range(5), c='yellow')
print(help(plt.legend))