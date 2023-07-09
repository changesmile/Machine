import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 利用jiebb分割字词，形成数组并返回
# def jieba_tokenize(text):
#     data = jieba.lcut(text)
#     return data


# tfidf_vect = TfidfVectorizer(tokenizer=jieba_tokenize, lowercase=False)
text_list = ["中国的小朋友高兴地跳了起来", "今年经济情况很好", "小明看起来很不舒服", "李小龙武功真厉害",
             "他很高兴去中国工作", "真是一个高兴的周末", "这件衣服太不舒服啦"]


# 聚类的文本集
# tfidf_matrix = tfidf_vect.fit(text_list)  # 训练
# print(tfidf_matrix.vocabulary_)  # 打印字典
# tfidf_matrix = tfidf_vect.transform(text_list)  # 转换
# tfidf_matrix = tfidf_vect.fit_transform(text_list)
# arr = tfidf_matrix.toarray()  # tfidf数组
# print('tfidf array:\n', arr)
# num_clusters = 4
# km = KMeans(n_clusters=num_clusters, max_iter=300, random_state=3)
#
# km.fit(tfidf_matrix)
# prt = km.predict(tfidf_matrix)
# print("Predicting result: ", prt)

def jieba_tokenizer(text):
    data = jieba.lcut(text)
    return data


tfidf_vect = TfidfVectorizer(tokenizer=jieba_tokenizer, lowercase=False)
tfidf_matrix = tfidf_vect.fit_transform(text_list)
arr = tfidf_matrix.toarray()
km = KMeans(n_clusters=4,max_iter=300,random_state=3)
km.fit(arr)
prt = km.predict(arr)
print(prt)