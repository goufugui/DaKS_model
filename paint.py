import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# from matplotlib.font_manager import _rebuild
# _rebuild()
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# sns.set(font='SimHei',font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小
print(matplotlib.matplotlib_fname())
matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体'STSong'仿宋

fig, ax = plt.subplots(figsize = (9,9))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
data = np.array([[0.1,0.2,0.3,0.3,0.1,0.2,0.3],[0.1,0.2,0.3,0.3,0.1,0.2,0.3],[0.1,0.2,0.3,0.3,0.4,0.5,0.6],[0.1,0.1,0.2,0.3,0.7,0.8,0.9],[0.1,0.2,0.3,0.3,0.1,0.2,0.3],[0.1,0.2,0.3,0.3,0.1,0.2,0.3]])
sns.heatmap(pd.DataFrame(data, index = range(1,7)),
                annot=False, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
ax.set_title('二维数组热力图',fontproperties='SimHei', fontsize = 18)
# ax.set_ylabel('数字', fontsize = 18)
# ax.set_xlabel('字母', fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
ax.set_yticklabels(['詹妮弗','毕业','院','校','是','耶鲁大学'], fontsize = 18, rotation = 360, horizontalalignment='right')
ax.set_xticklabels(['听说','她','还是','名校','的','高材生','呢'], fontsize = 18, horizontalalignment='right')
plt.show()