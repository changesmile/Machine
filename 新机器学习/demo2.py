# from sklearn.feature_extraction.text import CountVectorizer
#
# texts = ["dog cat fish", "dog cat cat", "fish bird", 'bird']
# cv = CountVectorizer()
# cv_fit = cv.fit_transform(texts)
#
# print(cv.get_feature_names_out())
# print(cv_fit)
#
# print(cv_fit.toarray().sum(axis=0))

from sklearn.feature_extraction.text import CountVectorizer

texts = ["dog cat fish bird", "dog cat cat", "fish bird", 'bird']
# 文本组合长度，ngram_range=(1, 4) 最多组合四种例如： "dog cat fish bird"
#             ngram_range=(1, 4) 最多组合两种例如： "dog cat" "cat fish"
cv = CountVectorizer(ngram_range=(1, 4))
cv_fit = cv.fit_transform(texts)

print(cv.get_feature_names_out())
print(cv_fit.toarray())

print(cv_fit.toarray().sum(axis=0))
