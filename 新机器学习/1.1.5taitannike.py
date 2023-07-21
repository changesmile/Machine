import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

titanic_data = pd.read_csv('./data/titanic.csv')
# describe = titanic_data.describe()
# print(describe)

# 年龄数据缺失，可以使用年龄的平均数填充
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())

# 将性别进行数值化
titanic_data.loc[titanic_data['Sex'] == 'male', 'Sex'] = 0
titanic_data.loc[titanic_data['Sex'] == 'female', 'Sex'] = 1

# 上船位置标记缺失，用最多的那个标记进行填充，然后进行数据化
titanic_data['Embarked'] = titanic_data['Embarked'].fillna('S')
titanic_data.loc[titanic_data['Embarked'] == 'S', 'Embarked'] = 0
titanic_data.loc[titanic_data['Embarked'] == 'C', 'Embarked'] = 1
titanic_data.loc[titanic_data['Embarked'] == 'Q', 'Embarked'] = 2

predictors = ['Pclass', 'Sex', 'Age','SibSp', 'Parch', 'Fare', 'Embarked']
alg = LogisticRegression()
kf = KFold(titanic_data.shape[0])