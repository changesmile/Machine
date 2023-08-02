import gensim
from gensim import corpora, models, similarities
import jieba

# jieba库会报一些信息
jieba.setLogLevel(jieba.logging.INFO)

punctuation = ["，", "。", "：", "；", "？"]
content = ["机器学习带动人工智能飞速的发展。",
           "深度学习带动人工智能飞速的发展。",
           "机器学习和深度学习带动人工智能飞速的发展。"
           ]
content_S = [jieba.lcut(text) for text in content]
content_S_Str = []
for line in content_S:
    words = []
    for word in line:
        if word in punctuation:
            continue
        words.append(word)
    content_S_Str.append(words)
dictionary = corpora.Dictionary(content_S_Str)
# 分词，将jieba中切分的词进行唯一id标注
# {'人工智能': 0, '发展': 1, '学习': 2, '带动': 3, '机器': 4, '的': 5, '飞速': 6, '深度': 7, '和': 8}
print(dictionary.token2id)
# 对已标注出唯一id的词，
# 可以使用doc2bow 字典的方法为文档创建词袋表示法，该方法返回单词计数的稀疏表示法
# 每个元组中的第一个条目对应于字典中令牌的ID，第二个条目对应于此令牌的计数。
# {'人工智能': 0, '发展': 1, '学习': 2, '带动': 3, '机器': 4, '的': 5, '飞速': 6, '深度': 7, '和': 8}
# ['机器', '学习', '带动', '人工智能', '飞速', '的', '发展']
# [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
corpus = [dictionary.doc2bow(text) for text in content_S_Str]
print(content_S_Str)
print(corpus)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
print(lda.print_topic(1, topn=5))
