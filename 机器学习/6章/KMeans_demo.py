from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#显示类别标签
print('k labels are:',kmeans.labels_)
#预测结果
print('predict results are:',kmeans.predict([[0, 0]]))
#显示簇中心
print('cluster centers are:',kmeans.cluster_centers_)