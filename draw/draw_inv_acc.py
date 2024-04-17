import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x_values = [5, 10, 15, 20]
bar_width = 0.15
num_of_values = 5
index = np.arange(len(x_values))
t20 = [1.81,8.86,10.89,4.15,1.83]
t15= [1.73,11.90,10.32,3.57,2.14]
t10 = [0.85,7.89,8.14,3.07,2.14]
t5 = [0.75,3.91,3.98,2.18,1.95]

# t20 = [88.5,85.0,85.4,78.0,72.0]
# t15= [88.5,85.0,85.4,77.0,71.0]
# t10 = [86.5,85.0,81.9,75.0,72.0]
# t5 = [86.5,79.0,76.1,73.0,65.0]

model_name = ['HAIformer', 'CoAD', 'MTDiag', 'PPO', 'DQN']
rgb1 = [(254/255,67/255,101/255), (252/255,157/255,154/255), (249/255,205/255,173/255), (200/255,200/255,160/255), (131/255,176/255,155/255)]
# 创建柱状图
for i in range(num_of_values):
    plt.bar(index + i * bar_width, [t5[i], t10[i], t15[i], t20[i]], bar_width, label=model_name[i], color = rgb1[i], linewidth= 0.5,edgecolor = 'k')
# plt.ylim(60, 100)
# 设置图表标题和标签
plt.xlabel(r'$\lambda_{max}$',fontsize=14)
plt.ylabel('IvI',fontsize=14)
# plt.ylabel('Acc (%)',fontsize=14)
plt.xticks(index + (bar_width * (num_of_values - 1)) / 2, x_values)
plt.legend()

plt.savefig('m_inv.png', dpi=500, bbox_inches='tight')
# plt.savefig('m_acc.png', dpi=500, bbox_inches='tight')
