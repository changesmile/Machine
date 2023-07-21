import pandas  # ipython notebook
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

titanic = pandas.read_csv("./data/titanic2.csv")


titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1


titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
# alg = LinearRegression()

# kf = KFold(n_splits=3, shuffle=False)
#
# predictions = []
# for train, test in kf.split(titanic):
#     # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
#     train_predictors = (titanic[predictors].iloc[train,:])
#     # The target we're using to train the algorithm.
#     train_target = titanic["Survived"].iloc[train]
#     # Training the algorithm using the predictors and target.
#     alg.fit(train_predictors, train_target)
#     # We can now make predictions on the test fold
#     test_predictions = alg.predict(titanic[predictors].iloc[test,:])
#     predictions.append(test_predictions)
#
# predictions = np.concatenate(predictions, axis=0)
#
# # Map predictions to outcomes (only possible outcomes are 1 and 0)
# predictions[predictions > .5] = 1
# predictions[predictions <=.5] = 0
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
alg = LinearRegression()
alg.fit(x_train, y_train)
score = alg.predict(x_test)
score[score > .5] = 1
score[score <= .5] = 0
accuracy = sum(score[score == titanic["Survived"]]) / len(score)
print(accuracy)