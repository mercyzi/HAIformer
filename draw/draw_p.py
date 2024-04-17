import matplotlib.pyplot as plt

# 生成x和y值
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = [0.53, 0.54, 0.57, 0.60, 0.61, 0.64, 0.63, 0.59, 0.57]


plt.plot(x, y, c=(254/255,67/255,101/255),linestyle='--',marker='v', markersize=7, linewidth=1)
# plt.plot(x, y2, c=(131/255,176/255,155/255),linestyle='--',marker='*', markersize=8, linewidth=1, label='MZ-4 dataset')
# plt.gca().invert_xaxis()
# plt.grid(axis='y', linestyle='-.', alpha=0.5)
# plt.ylim(75, 90)
plt.xlabel('Ratio')
plt.ylabel('MCC')
# plt.legend(loc = 'best')
plt.savefig('p5.png', dpi=500, bbox_inches='tight')


