import pandas as pd
from numpy import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances

# 1.读取数据
print('Step1:read data...')
food = pd.read_csv('E:/pythonProject/机器学习/datas/hot-spicy pot.csv')

# 2.将菜品的描述构造成TF-IDF向量
print('Step2:make TF-IDF...')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(food['taste'])
print(tfidf_matrix.shape)

# 3.计算两个菜品的余弦相似度
print('Step3:compute similarity...')


cosine_sim = pairwise_distances(tfidf_matrix, metric="cosine")


# 推荐函数，输出与其最相似的10个菜品
def content_based_recommendation(name):
    idx = indices[name]
    # 遍历 enumerate ，输出index 和 value
    # 再用list转化为列
    sim_scores = list(enumerate(cosine_sim[idx]))
    # lambad x:x[1] 取 sim_scores 的第二个值，也就是 cosine_sim 解析出来的
    sim_scores = sorted(sim_scores, key=lambda x: x[1])
    # 取出了第一个是自己以外的十个菜名
    sim_scores = sim_scores[1:11]
    # 取出菜名的index
    food_indices = [i[0] for i in sim_scores]
    return food['name'].iloc[food_indices]


# 4.根据菜名及特点进行推荐
print('Step4:recommend by name...')
# 建立索引，方便使用菜名进行数据访问
# 去重 drop_duplicates
indices = pd.Series(food.index, index=food['name']).drop_duplicates()
result = content_based_recommendation("celery")
print(result)
