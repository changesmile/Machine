import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

data = np.array([[1.1, 1.5, 1.4, 0.2],
                 [1.9, 1.0, 1.4, 0.2],
                 [1.7, 1.2, 1.3, 0.2],
                 [2.6, 2.1, 1.5, 0.2],
                 [2.0, 2.6, 1.4, 0.2]
                 ])
print(data)
knn = KNeighborsClassifier(1)
labels = ['A', 'B', 'C', 'D', 'F']
knn.fit(data, labels)
# reshape(1,-1)转化成1行
# 这是由于在sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列（比如前面做预测时，仅仅只用了一个样本数据），
# 所以需要使用numpy库的.reshape(1,-1)进行转换，而reshape的意思以及常用用法即上述内容
predict_num = knn.predict(np.array([[1.6, 1.5, 1.2, 0.1]]).reshape(1, -1))

print(predict_num)
