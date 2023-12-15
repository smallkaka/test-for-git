import cv2
import os
import re
from glob import glob
from skimage import data_dir, io, transform, color
import numpy as np
"""
记录一下自己改的脚本，太拉垮了，还有很多不足，对很多函数还不太懂
脚本的功能是将图片图片按中心裁剪成256*256的大小
备注：是一个(DAG4MIA)论文的代码的预处理
"""
#转化大小
def convert_size(f):
    rgb = io.imread(f)  # 依次读取rgb图片
    # gray = color.rgb2gray(rgb)  # 将rgb图片转换成灰度图
    dst = transform.resize(rgb, (256, 256))  # 将灰度图片大小转换为256*256
    return dst

save_path="/home/shiya.xu/papers/DAG4MIA/code/Data/REFUGE/Non-Glaucoma/resize_out/"
po_save = "/home/shiya.xu/papers/DAG4MIA/code/Data/REFUGE/Non-Glaucoma/image/*.png"
po_save_1 = "/home/shiya.xu/papers/DAG4MIA/code/Data/REFUGE/Non-Glaucoma/image/"
frames = glob(os.path.join(po_save_1, '*.png'))#读取所有图片

for i, frame in enumerate(frames):
    pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    data = pattern.findall(frame)
    data_now = data[0]#读取到了图片的原名称
    print(data_now)
    coll = io.ImageCollection(po_save, load_func=convert_size)
    io.imsave(save_path + data_now +'.png', coll[i])  # 循环保存图片


#   cv2.imwrite(video_path + str(data_now)+".png")
#   Newdir = os.path.join(video_path, str(data_now) + '.png')
#   img = cv2.imread(Newdir)
#   name = str(data_now) + ".png"
#   cv2.imwrite(dir + name, img)

