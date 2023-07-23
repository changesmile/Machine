from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# data = load_iris()
# print(data.keys())
# print(data['feature_names'])
# print(data['target'])
# x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=2)
# print(x_train, x_train.shape)
# print('-----')
# print(x_test, x_test.shape)
# print('-----')
# print(y_train, y_train.shape)
# print('-----')
# print(y_test, y_test.shape)
# df = pd.DataFrame(x_train, columns=data.feature_names)
# pd.plotting.scatter_matrix(df, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s = 60, alpha=0.8)
# plt.show()
# ----------------------------------------------
iris_data = load_iris()
iris_x = iris_data.data
iris_y = iris_data.target
# x_train 训练数据
# x_test  测试数据
# y_train 训练标签
# y_test  测试标签
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, random_state=0, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
knn = KNeighborsClassifier()
# fit 是对数据集进行训练
knn.fit(x_train, y_train)
#
result = knn.predict(x_test)
print('测试集大小：', x_test.shape)
print('真实测试结果：', y_test)
print('预测结果：   ', result)
# print(x_test)
# print()
# print(y_test)
print('预测精确度：', knn.result(x_test, y_test))


