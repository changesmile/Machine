import pandas as pd
import jieba

# jieba库会报一些信息
jieba.setLogLevel(jieba.logging.INFO)

def drop_stopwords(contents,stopwords):
    content_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(word)
        content_clean.append(line_clean)
    return content_clean, all_words


# 读取txt数据并标记标题
df_news = pd.read_table('./data/news/val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
# 去除空缺数据值得行
df_news = df_news.dropna()
content = df_news.content.values.tolist()
content_S = []
# 将val.txt中的content字段进行分词
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_S.append(current_segment)
# print(content_S[1000])
content_S = pd.DataFrame({'content_S':content_S})
# 分词后，对词进行停词，对没有用或者不需要的词筛选掉
stopwords = pd.read_csv("./data/news/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'],
                        encoding='utf-8')
contents = content_S.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

df_content=pd.DataFrame({'contents_clean':contents_clean})
df_content.head()