import collections
import re


def words(text):
    # 正则表达式，转换成小写，搜索单词，
    data_dict = re.findall('[a-z]+', text.lower())
    return data_dict


def train(features):
    # defaultdict ： 当字典查询时，为key不存在提供一个默认值。
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


NWORDS = train(words(open('./data/big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):  # 一位纠错
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion


def known_edits2(word):  # 两位纠错
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words):  # 检查字符串是否在词库种存在，存在则保存
    data = set(w for w in words if w in NWORDS)
    return data


def correct(word):
    # known([word]) 检查原始字符串是否在词库中
    # known(edits1(word)) 检查一位纠错情况下的字符串是否在词库中
    # known_edits2(word)  检查在一位纠错情况下再次纠错的字符串是否在词库中(两位纠错)
    # 都没有则返回原始字符串
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=lambda w: NWORDS[w])


print(correct('knon'))
