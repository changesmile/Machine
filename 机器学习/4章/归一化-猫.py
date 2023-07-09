import pandas as pd


def MaxMinNormalization(x):
    headers = ['Lwsk/mm', 'LEar/mm', 'Weight/g']
    result = pd.DataFrame(columns=headers)
    # 原函数直接用下标，这里直接取每行的值
    # 再把每行的值和标题头挨个取出
    for index, row in x.loc[:, ['Lwsk/mm', 'LEar/mm', 'Weight/g']].iterrows():
        dict1 = {}
        for i, header in zip(row, headers):
            maxCol = x[header].max()
            minCol = x[header].min()
            val = (i - minCol) / (maxCol - minCol)
            dict1[header] = val
        result = result._append(dict1, ignore_index=True)
    result['No'] = x['No']
    result['color'] = x['color']
    return result


data1 = pd.read_csv('../datas/CatInfo.csv')
# 先删除有空值的行
data1 = data1.dropna(how='any')
data1 = data1.reset_index(drop=True)
# # 创建一个空表保存No color 字段
# data2 = pd.DataFrame(columns=['No', 'color'])
# data2['No'] = data1['No']
# data2['color'] = data1['color']
# # 删除两个用例数据
# del data1['No']
# del data1['color']
# # 再把每个用例的值转换为int类型
# headers = list(data1)
for header in ['Lwsk/mm', 'LEar/mm', 'Weight/g']:
    data1[header] = data1[header].astype(int)

newData = MaxMinNormalization(data1)
print(newData)
