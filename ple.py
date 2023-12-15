from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


img=np.array(cv2.imread('/home/shiya.xu/Api/test_images/1_1.png',cv2.IMREAD_GRAYSCALE)) #打开图像并转化为数字矩阵


pos = np.unravel_index(np.argmax(img),img.shape)
print('像素最大值位置:\n',pos)
pos_min = np.unravel_index(np.argmin(img),img.shape)
print('像素最小值位置:\n',pos_min)

print ('像素矩阵大小:\n',img.shape)
# print ('zhi:\n',len(img.shape))
print ('像素矩阵:\n',img)
print('像素最大值:\n',img[pos[0]][pos[1]])
print('像素最小值:\n',img[pos_min[0]][pos_min[1]])

plt.figure('cat')
plt.imshow(img)
plt.axis('on')
plt.show()
