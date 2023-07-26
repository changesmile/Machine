import re
import pandas as pd  # ipython notebook
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt
# 忽略警告提示
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
full = train.append(test, ignore_index=True)
# print(full.shape)
# print(full.describe())
# 年龄(Age)
full['Age'] = full['Age'].fillna(full['Age'].mean())
# 船票价格(Fare)
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
full['Embarked'] = full['Embarked'].fillna('S')
sex_mapDict = {'male': 1, 'female': 0}
# map函数：对Series每个数据应用自定义的函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)

embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')
full = pd.concat([full, embarkedDf], axis=1)
full.drop('Embarked', axis=1, inplace=True)
pclassDf = pd.get_dummies(full['Pclass'], prefix='Pclass')
full = pd.concat([full, pclassDf], axis=1)
full.drop('Pclass', axis=1, inplace=True)
print(full.head())
