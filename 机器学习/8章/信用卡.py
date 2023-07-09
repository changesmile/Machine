from sklearn.linear_model import LogisticRegression
import numpy as np

data = np.array([[20, 7000, 800, 1], [35, 2000, 2500, 0], [27, 5000, 3000, 1], [32, 4000, 4000, 0], [45, 2000, 3800, 0]])
X_train = data[:, :3]
y_train = data[:, 3]
test_data = np.array([[34, 3500, 3500]])

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
predict_data = clf.predict(test_data)
print(predict_data)