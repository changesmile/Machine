import gensim
from gensim import corpora, models, similarities
import jieba
from sklearn.feature_extraction.text import CountVectorizer

# jieba库会报一些信息
jieba.setLogLevel(jieba.logging.INFO)
# 停词表
punctuation = ["，", "。", "：", "；", "？"]
# 需要分析的词
content = ["机器学习带动人工智能飞速的发展。",
           "深度学习带动人工智能飞速的发展。",
           "机器学习和深度学习带动人工智能飞速的发展。"
           ]
# 使用jieba进行切分
content_S = [jieba.lcut(text) for text in content]
content_S_Str = []
# 把切分的词用空格拼接起来
for content in content_S:
    content = " ".join(content)
    content_S_Str.append(content)
# content_S = [' '.join(jieba.lcut(text)) for text in content]  # 一步到位
# 使用 CountVectorizer 加入停用词
vec = CountVectorizer(stop_words=punctuation)
# 转化为词频统计表
train = vec.fit_transform(content_S_Str)
print(content_S)
# 输出词频的词根
print(vec.get_feature_names_out())
# 词频的唯一标识符
print(vec.vocabulary_)
# 转化为稀疏矩阵， 统计每段分析词出现词根的数量
print(train.toarray())
