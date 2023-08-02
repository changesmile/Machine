# 语料库
import gensim
import jieba

# jieba库会报一些信息
jieba.setLogLevel(jieba.logging.INFO)
from gensim import corpora

punctuation = ["，", "。", "：", "；", "？"]
content = ["机器学习带动人工智能飞速的发展。",
           "深度学习带动人工智能飞速的发展。",
           "机器学习和深度学习带动人工智能飞速的发展。"
           ]

segs_1 = [jieba.lcut(con) for con in content]
tokenized = []
for sentence in segs_1:
    words = []
    for word in sentence:
        if word not in punctuation:
            words.append(word)
    tokenized.append(words)
# print("\n去除停止词后：",tokenized)
dictionary = corpora.Dictionary(tokenized)
# 输出转换成词根对应的唯一id
print(dictionary.token2id)
new_doc = "人机交互和人工智能带来重大变革"
segs_new = jieba.lcut(new_doc)
# print(segs_new)
new_vec = [dictionary.doc2bow(segs_new)]
bow_corpus = [dictionary.doc2bow(text) for text in tokenized]
# dictionary.doc2bow 转化为向量
print(bow_corpus)
lda = gensim.models.ldamodel.LdaModel(corpus=new_vec, id2word=dictionary, num_topics=20)
lda2 = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=20)
print(lda.print_topic(1))
print(lda2.print_topic(1))
