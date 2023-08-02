import pandas as pd
import jieba

# jieba库会报一些信息
jieba.setLogLevel(jieba.logging.INFO)
def drop_stopwords(contents, stopwords):
    line_clean = []
    for word in contents:
        if word in stopwords:
            continue
        line_clean.append(word)
    return line_clean

df_news = pd.read_table('./data/news/val.txt', names=['cate', 'theme', 'url', 'content'], encoding='utf-8')
df_news = df_news.dropna()
content = df_news.content.values.tolist()[0]
content_S = jieba.lcut(content)
stopwords = pd.read_csv('./data/news/stopwords.txt', quoting=3, sep='\t', index_col=False, encoding='utf-8', names=['stopword'])
stopwords = stopwords.stopword.values.tolist()
contents_clean = drop_stopwords(content_S, stopwords)
print(contents_clean)