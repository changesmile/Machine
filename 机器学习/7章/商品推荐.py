from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
from numpy import *
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    data = pd.read_csv('E:/pythonProject/机器学习/datas/UserInfo.csv')
    data = data.fillna(0)
    # print(data.iloc[:, 1:])
    # scaler = MinMaxScaler(feature_range=(0, 1))  # 数据标准化
    # dataset = scaler.fit_transform(data)
    similarity = pairwise_distances(data, metric="euclidean")
    # print(similarity)
    recmdStr = ''
    label = pd.read_csv('E:/pythonProject/机器学习/datas/userFavorit.csv').values.tolist()
    for m in range(similarity.shape[0]):
        recmd = ''
        a = '为用户' + str(m + 1) + '推荐:'
        simMax = 0
        for n in range(similarity.shape[1]):
            if simMax < similarity[m][n] and similarity[m][n] != 1:  # 取非本人的最相似用户
                simMax = similarity[m][n]
                recmd = label[n - 1]
            else:
                continue
        a = a + str(recmd) + "\n"
        recmdStr += a
    print(recmdStr)
