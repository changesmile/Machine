import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

data = pd.read_csv('E:/pythonProject/机器学习/datas/Customer_Info.csv')
X = data.iloc[:, [4, 3]].values
# print(X)
# sumCenter = []
#
# for i in range(1,11):
#     knn = KMeans(n_clusters=i, n_init='auto').fit(X)
#     numCenter = knn.inertia_
#     print(numCenter)
#     sumCenter.append(numCenter)
# plt.plot(range(1, 11), sumCenter)
# plt.show()
knn = KMeans(n_clusters=4, n_init=10, random_state=0, max_iter=300)
y_kmeans = knn.fit_predict(X)
# print(y_kmeans.shape)
# print(X[y_kmeans == 0, 0])
# print(X[y_kmeans == 0, 1])
# print(X[y_kmeans == 2, 0])
# print(X[y_kmeans == 2, 1])
# print(X[y_kmeans == 1, 0])
# print(X[y_kmeans == 1, 1])
numdict = [0, 0, 0, 0]
for i in y_kmeans:
    numdict[i] = numdict[i] + 1
print(numdict)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, marker='^', c='red', label='No rich')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, marker='o', c='green', label='Middle_down')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, marker='*', c='black', label='Middle_up')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, marker='*', c='blue', label='Rich')
plt.scatter(knn.cluster_centers_[:, 0], knn.cluster_centers_[:, 1], s=250, c='yellow', label='Center')
plt.legend()
plt.xlabel('Deposit')
plt.ylabel('age')
plt.show()
