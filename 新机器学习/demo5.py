from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

iris = load_iris()
logreg = LogisticRegression()

# scores = cross_val_score(logreg, iris.data, iris.target)
# print("Cross-validation scores: {}".format(scores))

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Cross-validation scores: {}".format(scores))

