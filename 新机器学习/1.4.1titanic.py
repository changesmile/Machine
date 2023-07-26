import re
import pandas  # ipython notebook
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

# titanic = pandas.read_csv("./data/titanic.csv")
#
# titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
# titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
# titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#
#
# titanic["Embarked"] = titanic["Embarked"].fillna('S')
# titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
# titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
# titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#
# titanic.to_csv('./data/titanic2.csv')

# ------------------------------
titanic = pandas.read_csv("./data/titanic2.csv")
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LinearRegression()

kf = KFold(n_splits=3, shuffle=False)

predictions = []
for train, test in kf.split(titanic):
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)

# alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
# kf = KFold(n_splits=3)
# scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
# print(scores.mean())

# ---------------------------------------
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
titanic['NameLength'] = titanic['Name'].apply(lambda x: len(x))


def get_title(name):
    # 匹配除换行符 \n之外的任何单字符。要匹配. ，请使用 \.
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''


titles = titanic['Name'].apply(get_title)
# print(pandas.value_counts(titles))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v

titanic['Title'] = titles
#
# predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]
#
# selector = SelectKBest(f_classif, k=5)
# selector.fit(titanic[predictors], titanic['Survived'])
# scores = -np.log10(selector.pvalues_)
# plt.figure()
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

# ------------------------------
# algorithms = [
#     [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title",]],
#     [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
# ]
#
# # Initialize the cross validation folds
# kf = KFold(n_splits=3, random_state=42, shuffle=True)
#
# predictions = []
# for train, test in kf.split(titanic):
#     train_target = titanic["Survived"].iloc[train]
#     full_test_predictions = []
#     for alg, predictors in algorithms:
#         alg.fit(titanic[predictors].iloc[train,:], train_target)
#         test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
#         full_test_predictions.append(test_predictions)
#     test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
#     test_predictions[test_predictions <= .5] = 0
#     test_predictions[test_predictions > .5] = 1
#     predictions.append(test_predictions)
#
# predictions = np.concatenate(predictions, axis=0)
# accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
# print(accuracy)

# alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
# # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
# kf = KFold(n_splits=3)
# scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
#
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())

x_train, x_test, y_train, y_test = train_test_split(titanic[predictors], titanic['Survived'],test_size=0.3 , random_state=0)
model = LogisticRegression()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)