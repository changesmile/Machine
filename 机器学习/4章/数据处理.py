import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(root_directory)
x = np.array([[3, -2, 490],
              [3, 0.5, 520],
              [1, 2, -443]
              ])
# 数据标准化 正态分布
x_scaled = preprocessing.scale(x)
# print(x_scaled)
print('----------------')
# 数据缩放
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)
# print(x_minmax)
print('----------------')

# 数据拆分
data = pd.read_csv('E:/pythonProject/机器学习/datas/CatInfo.csv')
cat_train_X, cat_test_X, cat_train_y, cat_test_y = train_test_split(data['Lwsk/mm'],data['LEar/mm'], test_size=0.3,random_state=0)
print(cat_train_X)
print(cat_test_X)
print(cat_train_y)
print(cat_test_y)



