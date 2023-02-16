
##  基础函数库
import numpy as np
import pandas as pd
## 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data1.csv')
data.info()
data.head()
data.describe()
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(15,12),dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.xticks(rotation = 45,fontsize=12)
plt.yticks(fontsize=12)  # 设置坐标标签字体大小
plt.savefig( 'reli.png')
plt.show()