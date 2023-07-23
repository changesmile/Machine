import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
import re
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

train_data = pd.read_csv('D:/pythonProject/MachineLearn/新机器学习/taitannik/train.csv')
test_data = pd.read_csv('D:/pythonProject/MachineLearn/新机器学习/taitannik/test.csv')

full = pd.concat([train_data, test_data], ignore_index=True)
# 查看数据信息，有缺失的部分
# print(full.describe())
full['Age'] = full['Age'].fillna(full['Age'].median())  # 年龄缺失取均值
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())  # 票价缺失取均值

# 登船港口数据有缺失，以登船最多的港口补全
# print(full['Embarked'].value_counts())
full['Embarked'] = full['Embarked'].fillna('S')


def ChangeSexNum(name):  # 将性别转换成数值
    if name == 'male':
        return 0
    elif name == 'female':
        return 1


def ChangeEmbarkedNum(Embarked):
    if Embarked == 'S':
        return 0
    elif Embarked == 'C':
        return 1
    elif Embarked == 'Q':
        return 2


def ChangeFamilyNum(FamilySize):
    if FamilySize == 1:
        return 0
    if 2 <= FamilySize <= 4:
        return 1
    if 5 <= FamilySize:
        return 2


def ChangeNameTitle(name):
    title_name_search = re.search(' ([A-Za-z]+)\.', name)
    if title_name_search:
        return title_name_search.group(0).split('.')[0]
    return ''


full['Sex'] = full['Sex'].apply(ChangeSexNum)
full['Embarked'] = full['Embarked'].apply(ChangeEmbarkedNum)
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['FamilySize'] = full['FamilySize'].apply(ChangeFamilyNum)
full['TitleName'] = full['Name'].apply(ChangeNameTitle)
# 输出头衔种类 总共18种头衔
# print(full['TitleName'].unique())
title_dict = {' Mr': 0,
              ' Mrs': 1,
              ' Miss': 2,
              ' Master': 3,
              ' Don': 4,
              ' Rev': 5,
              ' Dr': 6,
              ' Mme': 7,
              ' Ms': 8,
              ' Major': 9,
              ' Lady': 10,
              ' Sir': 11,
              ' Mlle': 12,
              ' Col': 13,
              ' Capt': 14,
              ' Countess': 15,
              ' Jonkheer': 16,
              ' Dona': 17}
full['TitleName'] = full['TitleName'].map(title_dict)
full.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)
sourceRow = 891
score_x = full.loc[:sourceRow - 1, full.columns != 'Survived']
score_y = full.loc[:sourceRow - 1, full.columns == 'Survived']
pred_x = full.loc[sourceRow:, full.columns != 'Survived']
pred_y = pd.read_csv('D:/pythonProject/MachineLearn/新机器学习/taitannik/gender_submission.csv')

train_X, test_X, train_y, test_y = train_test_split(score_x, score_y, train_size=.8)
model = LogisticRegression()
model.fit(train_X, train_y)
predict_data = model.predict(pred_x)
# predict_data = pd.DataFrame(predict_data, columns=["Survived"])
# predict_data["Survived"] = predict_data["Survived"].astype('int')
# score = len(pred_y[predict_data["Survived"] == pred_y["Survived"]]) / len(pred_y)
score = model.score(test_X, test_y)
print(score)
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
     ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "TitleName", ]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "TitleName", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(n_splits=3, random_state=1, shuffle=True)
titanic = full.loc[:sourceRow - 1]
predictions = []
for train, test in kf.split(titanic):
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train, :], train_target)
        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == full["Survived"]]) / len(predictions)
print(accuracy)
