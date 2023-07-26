from sklearn.datasets import fetch_california_housing
from sklearn import tree
from IPython.display import Image, display
import pydotplus
import matplotlib.pyplot as plt

# import os
# os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

dtr = tree.DecisionTreeRegressor(max_depth=2)
housing = fetch_california_housing()
dtr.fit(housing.data[:, [6, 7]], housing.target)
# print(housing.data)
# print(housing.target)
dot_data = \
    tree.export_graphviz(
        dtr,
        out_file=None,
        feature_names=housing.feature_names[6:8],
        filled=True,
        impurity=False,
        rounded=True
    )

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")

graph.write_png("./dtr_white_background.png")
