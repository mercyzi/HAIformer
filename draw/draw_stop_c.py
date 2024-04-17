
import numpy as np
import matplotlib.pyplot as plt


    

x = [0.94, 0.92, 0.90, 0.88, 0.86, 0.84]

y1 = [88.46, 88.46, 88.46, 88.46, 87.5, 87.5]
y2 = [78.52,78.52,78.17,78.17,78.00,77.94]

# y1 = [2.8269,
# 2.7692,
# 2.6442,
# 2.3462,
# 1.9712,
# 1.6154]
# y2= [1.3732,
# 1.3803,
# 1.0775,
# 1.0,
# 0.9859,
# 0.9225]

# y1= [0.2981,
# 0.2788,
# 0.2596,
# 0.2308,
# 0.2115,
# 0.1827]
# y2= [0.1479,
# 0.1408,
# 0.1338,
# 0.1127,
# 0.1056,
# 0.1050]
plt.figure(figsize=(6, 3))
plt.plot(x, y1, c=(254/255,67/255,101/255),linestyle='--',marker='v', markersize=7, linewidth=1, label='Dxy dataset')
plt.plot(x, y2, c=(131/255,176/255,155/255),linestyle='--',marker='*', markersize=8, linewidth=1, label='MZ-4 dataset')
plt.gca().invert_xaxis()
plt.grid(axis='y', linestyle='-.', alpha=0.5)
plt.ylim(75, 90)
plt.xlabel(r'$\delta_d$')
plt.ylabel('Acc')
plt.legend(loc = 'best')
plt.savefig('acc.png', dpi=500, bbox_inches='tight')


