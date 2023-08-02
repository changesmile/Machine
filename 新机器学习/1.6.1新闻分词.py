import pandas as pd
import jieba
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# jieba库会报一些信息
jieba.setLogLevel(jieba.logging.INFO)


def drop_stopwords(contents, stopwords):
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
# 打乱随机顺序
# df_news = sklearn.utils.shuffle(df_news)
df_news = df_news.sample(frac=1, random_state=42)
# 去除空缺数据值得行
df_news = df_news.dropna()
content = df_news.content.values.tolist()[:1000]
content_S = []
# 将val.txt中的content字段进行分词
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 2 and current_segment != '\r\n':
        content_S.append(current_segment)
# print(content_S[1000])
content_S = pd.DataFrame({'content_S': content_S})
# 分词后，对词进行停词，对没有用或者不需要的词筛选掉
stopwords = pd.read_csv("./data/news/stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'],
                        encoding='utf-8')
contents = content_S.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean, all_words = drop_stopwords(contents, stopwords)

# print(df_content.iloc[0])
df_train = pd.DataFrame({'contents_clean': contents_clean, 'label': df_news['category'][:1000]})
df_train = df_train.dropna()
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9,
                 "时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping)
x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values)
# print(x_test)
words = []
for line_index in range(len(x_train)):
    try:
        # x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index)
# lowercase将所有字符转变为小写
# CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。
# 对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
vec.fit(words)

classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)

test_words = []
for line_index in range(len(x_test)):
    try:
        # x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
        print(line_index)
test_score = classifier.score(vec.transform(test_words), y_test)
print(test_score)

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
vectorizer.fit(words)
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
test_score = classifier.score(vectorizer.transform(test_words), y_test)
print(test_score)
