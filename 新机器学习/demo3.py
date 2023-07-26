#【例3.76】使用停用词，对文本进行分词。
import jieba
import jieba.analyse

#stop-words list
def stopwordslist(filepath):
    f=open(filepath,'r',encoding='utf-8')
    txt=f.readlines()
    stopwords=[]
    for line in txt:
        stopwords.append(line.strip())
    return stopwords

inputs=open('news.txt','rb')
stopwords=stopwordslist('ch-stop_words.txt')
outstr=''
for line in inputs: 
    sentence_seged=jieba.cut(line.strip())    
    for word in sentence_seged:        
        if word not in stopwords:
            if word!='\t':
                outstr+=' '+word
                outstr+=''
print(outstr)