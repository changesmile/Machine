import jieba.posseg as pseg
seg_list = pseg.cut('今天我终于看到了南京长江大桥')
print(seg_list)
for w, t in seg_list:
    print(w+'  '+t)
# result = ' '.join(['{0}/{1}'.format(w,t) for w,t in seg_list])
# print(result)