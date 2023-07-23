import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# 忽略警告提示
import warnings
warnings.filterwarnings('ignore')

# 读入训练集和测试集及其标签：
train = pd.read_csv("./train.csv")
# 测试数据集
test = pd.read_csv("./test.csv")
# 这里要记住训练数据集有891条数据，方便后面从中拆分出测试数据集用于提交Kaggle结果
# 合并数据集，方便同时对两个数据集进行清洗
full = pd.concat([train, test], ignore_index=True)
# 年龄(Age)
full['Age'] = full['Age'].fillna(full['Age'].mean())
# 船票价格(Fare)
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
full['Embarked'] = full['Embarked'].fillna('S')
sex_mapDict = {'male': 1,
               'female': 0}
# map函数：对Series每个数据应用自定义的函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)
embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')
# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, embarkedDf], axis=1)
full.drop('Embarked', axis=1, inplace=True)
# 使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies(full['Pclass'], prefix='Pclass')
# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, pclassDf], axis=1)

# 删掉客舱等级（Pclass）这一列
full.drop('Pclass', axis=1, inplace=True)


# # 练习从字符串中提取头衔，例如Mr
# # split用于字符串分割，返回一个列表
# # 我们看到姓名中'Braund, Mr. Owen Harris'，逗号前面的是“名”，逗号后面是‘头衔. 姓’
# name1 = 'Braund, Mr. Owen Harris'
# '''
# split用于字符串按分隔符分割，返回一个列表。这里按逗号分隔字符串
# 也就是字符串'Braund, Mr. Owen Harris'被按分隔符,'拆分成两部分[Braund,Mr. Owen Harris]
# 你可以把返回的列表打印出来瞧瞧，这里获取到列表中元素序号为1的元素，也就是获取到头衔所在的那部分，即Mr. Owen Harris这部分
# '''
# # Mr. Owen Harris
# str1 = name1.split(',')[1]
# '''
# 继续对字符串Mr. Owen Harris按分隔符'.'拆分，得到这样一个列表[Mr, Owen Harris]
# 这里获取到列表中元素序号为0的元素，也就是获取到头衔所在的那部分Mr
# '''
# # Mr.
# str2 = str1.split('.')[0]
# # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
# str3 = str2.strip()
# '''
# 定义函数：从姓名中获取头衔
# '''


def getTitle(name):
    str1 = name.split(',')[1]  # Mr. Owen Harris
    str2 = str1.split('.')[0]  # Mr
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3 = str2.strip()
    return str3


# 存放提取后的特征
titleDf = pd.DataFrame()
# map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = full['Name'].map(getTitle)
# 姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}

# map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

# 使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
full = pd.concat([full, titleDf], axis=1)

# 删掉姓名这一列
full.drop('Name', axis=1, inplace=True)
# sum = lambda a, b: a + b
# '''
# 客场号的类别值是首字母，例如：
# C85 类别映射为首字母C
# '''
# full['Cabin'] = full['Cabin'].map(lambda c: c[0])
#
# ##使用get_dummies进行one-hot编码，列名前缀是Cabin
# cabinDf = pd.get_dummies(full['Cabin'], prefix='Cabin')
# # 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
# full = pd.concat([full, cabinDf], axis=1)
#
# # 删掉客舱号这一列
# full.drop('Cabin', axis=1, inplace=True)
# 存放家庭信息
familyDf = pd.DataFrame()

'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1

'''
家庭类别：
小家庭Family_Single：家庭人数=1
中等家庭Family_Small: 2<=家庭人数<=4
大家庭Family_Large: 家庭人数>=5
'''
# if 条件为真的时候返回if前面内容，否则返回0
familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s: 1 if s == 1 else 0)
familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
# 添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full, familyDf], axis=1)
full.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
# 相关性矩阵
corrDf = full.corr()
print(corrDf)
full_X = pd.concat([titleDf,  # 头衔
                    pclassDf,  # 客舱等级
                    familyDf,  # 家庭大小
                    full['Fare'],  # 船票价格
                    embarkedDf,  # 登船港口
                    full['Sex']  # 性别
                    ], axis=1)
# 原始数据集有891行
sourceRow = 891

'''
sourceRow是我们在最开始合并数据前知道的，原始数据集有总共有891条数据
从特征集合full_X中提取原始数据集提取前891行数据时，我们要减去1，因为行号是从0开始的。
'''
# 原始数据集：特征
source_X = full_X.loc[0:sourceRow - 1, :]
# 原始数据集：标签
source_y = full.loc[0:sourceRow - 1, 'Survived']

# 预测数据集：特征
pred_X = full_X.loc[sourceRow:, :]
'''
确保这里原始数据集取的是前891行的数据，不然后面模型会有错误
'''
# 原始数据集有多少行
print('原始数据集有多少行:', source_X.shape[0])
# 预测数据集大小
print('原始数据集有多少行:', pred_X.shape[0])
# 原始数据集有多少行: 891
# 原始数据集有多少行: 418
'''
从原始数据集（source）中拆分出训练数据集（用于模型训练train），测试数据集（用于模型评估test）
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和test data
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
'''

# 建立模型用的训练数据集和测试数据集
train_X, test_X, train_y, test_y = train_test_split(source_X,
                                                    source_y,
                                                    train_size=.8)

model = LogisticRegression()
model.fit(train_X, train_y)

score = model.score(test_X, test_y)
print(score)
