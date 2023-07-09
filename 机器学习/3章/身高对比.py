import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False
df = pd.read_csv("../datas/avgHgt.csv")
x_values = range(7, 19)
plt.xticks(range(7, 19))
plt.plot(x_values, df['CHeight'], label='中国男孩身高')
plt.plot(x_values, df['JHeight'], label='日本男孩身高')
plt.title('中日两国7-12岁男孩身高图')
plt.xticks(rotation=45)
plt.xlabel('年龄/岁')
plt.ylabel('身高/厘米')
plt.legend()
plt.show()
