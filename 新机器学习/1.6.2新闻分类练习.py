import pandas as pd
import jieba
import jieba.analyse

import numpy

# jieba库会报一些信息
jieba.setLogLevel(jieba.logging.INFO)


def drop_stopwords(contents, stopwords):
    line_clean = []
    for word in contents:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean


# 读取txt数据并标记标题
df_news = pd.read_table('./data/news/val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
# 去除空缺数据值得行
df_news = df_news.dropna()
content = df_news.content.values.tolist()[0]
content_S = []
# 将val.txt中的content字段进行分词

current_segment = jieba.lcut(content)
stopwords = pd.read_csv("./data/news/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'],
                        encoding='utf-8')
stopwords = stopwords.stopword.values.tolist()
contents_clean = drop_stopwords(current_segment, stopwords)
df_all_words = pd.DataFrame(contents_clean, columns=['all_words'])
words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count"})
# 升序降序 ascending
words_count = words_count.reset_index().sort_values(by=['count'], ascending=False)
# print(words_count)
content_str = df_news.content.values.tolist()[240]
print(''.join(content_str))
content_str_S = ' '.join(jieba.analyse.extract_tags(content_str, topK=5,  withWeight=False))
print(content_str_S)