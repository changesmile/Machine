import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruit = pd.read_csv('E:/pythonProject/机器学习/datas/fruit_data.txt', sep="\t")
# 获取属性
X = fruit.iloc[:, 1:]
# 获取类别
Y = fruit.iloc[:, 0].T

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(1)
knn.fit(train_x, train_y)
result = knn.predict(test_x)
print('数据大小：', fruit.shape)
print('测试大小', test_x.shape)
print('真实结果：', test_y)
print('测试结果：', result)
score = knn.result(test_x, test_y)
print('精确率：', score)


