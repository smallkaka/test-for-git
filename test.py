import numpy as np





a = np.random.randint(10, 100, size=9)
a = a.reshape((3,3))
print(a)

# 第2大数值
max2 = np.sort(a)[-2]
print(max2)
# 第2大索引
max_index2 = np.argsort(a)[-2]
print(max_index2)
