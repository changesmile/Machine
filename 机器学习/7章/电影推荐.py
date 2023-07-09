# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# user-based/item-based预测函数
def predict(scoreData, similarity, type='user'):
    # 1. 基于物品的推荐
    predt_Mat = ''
    if type == 'item':
        # 评分矩阵scoreData乘以相似度矩阵similarity，再除以相似度之和
        # numpy.dot() 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和
        # a = np.array([[1,2],[3,4]])
        # b = np.array([[11,12],[13,14]])
        # print(np.dot(a,b))  [[1*11+2*13, 1*12+2*14],         [[37  40]
        #                      [3*11+4*13, 3*12+4*14]]          [85  92]]
        # numpy.abs() 取绝对值
        predt_Mat = scoreData.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    elif type == 'user':
        # 2. 基于用户的推荐
        # 计算用户评分均值，减少用户评#分高低习惯影响
        user_meanScore = scoreData.mean(axis=1)
        # reshape(1,-1)转化成1行
        # 这是由于在sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列（比如前面做预测时，仅仅只用了一个样本数据），
        # 所以需要使用numpy库的.reshape(1,-1)进行转换，而reshape的意思以及常用用法即上述内容
        score_diff = (scoreData - user_meanScore.reshape(-1, 1))  # 获得评分差值
        # 推荐结果predt_Mat: 等于相似度矩阵similarity乘以评分差值矩阵
        # score_diff，再除以相似度之和，最后加上用户评分均值user_meanScore。
        # similarity 已经经过余弦相似度算法得到向量值
        predt_Mat = user_meanScore.reshape(-1, 1) + similarity.dot(score_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    return predt_Mat


# 步骤1.读数据文件
print('step1.Loading dataset...')
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
scoreData = pd.read_csv('E:/pythonProject/机器学习/datas/u.data', sep='\t', names=r_cols, encoding='latin-1')
print('  scoreData shape:', scoreData.shape)

# 步骤2.生成用户-物品评分矩阵
print('step2.Make user-item matrix...')
n_users = 943
n_items = 1682
data_matrix = np.zeros((n_users, n_items))
for line in range(np.shape(scoreData)[0]):
    row = scoreData['user_id'][line] - 1
    col = scoreData['movie_id'][line] - 1
    score = scoreData['rating'][line]
    data_matrix[row, col] = score
print('  user-item matrix shape:', data_matrix.shape)

# 步骤3.计算相似度
print('step3.Computing similarity...')
# 使用pairwise_distances函数，简单计算余弦相似度
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')  # T转置转变计算方向
print('  user_similarity matrix shape:', user_similarity.shape)
print('  item_similarity matrix shape:', item_similarity.shape)

# 步骤4.使用相似度进行预测
print('step4.Predict...')
user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')
print('ok.')

# 步骤5 显示推荐结果
print('step5.Display result...')
print('------------------------')
print('(1)UBCF predict shape', user_prediction.shape)
print('  real answer is:\n', data_matrix[:5, :5])
print('  predict result is:\n', user_prediction[:5, :5])
print('(2)IBCF predict shape', item_prediction.shape)
print('  real answer is:\n', data_matrix[:5, :5])
print('  predict result is:\n', item_prediction[:5, :5])
