import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = [0.9, 0.7, 0.5, 0.3, 0.1]

#dxy: 9:0.902 ; 7:0.8962 ; 5: ; 3: ; 1: 0.9

y1 = [0.8854, 0.8788, 0.8788, 0.880, 0.883] 
y2 = [0.9113, 0.9121, 0.9092, 0.9100, 0.9100 ]
y1 = [i*100 for i in y1]
y2 = [i*100 for i in y2]
rgb1 = [(254/255,67/255,101/255), (252/255,157/255,154/255), (249/255,205/255,173/255), (200/255,200/255,160/255), (131/255,176/255,155/255)]
# 创建柱状图
# for i in range(num_of_values):
#     plt.bar(index + i * bar_width, [t5[i], t10[i], t15[i], t20[i]], bar_width, label=model_name[i], color = rgb1[i], linewidth= 0.5,edgecolor = 'k')
plt.axhline(y=91.21, color = rgb1[3], linestyle='-.', linewidth=1,label='Doctor (MDD)')
plt.axhline(y=88.54, color = rgb1[1], linestyle='-.', linewidth=1,label='Doctor (Dxy)')

plt.plot(x, y2, color = rgb1[4], marker='*', markersize=8,label='HAIformer (MDD) ')
plt.plot(x, y1, color = rgb1[0], marker='v', markersize=7,label='HAIformer (Dxy)')
plt.gca().invert_xaxis()
plt.ylim(87.6, 91.5)
# 设置图表标题和标签
plt.xlabel(r'$\gamma$',fontsize=14)
plt.ylabel('Acc (%)',fontsize=14)
plt.xticks([0.9, 0.7, 0.5, 0.3, 0.1])
# plt.xticks(index + (bar_width * (num_of_values - 1)) / 2, x_values)
plt.legend()

plt.savefig('weight.png', dpi=500, bbox_inches='tight')
# plt.savefig('m_acc.png', dpi=500, bbox_inches='tight')
