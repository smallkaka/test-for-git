import pydicom
import matplotlib.pyplot as plt
import scipy.misc
import pandas as pd
import numpy as np
import os
import imageio


def Dcm2jpg(file_path):
    # 获取所有图片名称
    c = []
    names = os.listdir(file_path)  # 路径
    # 将文件夹中的文件名称与后边的 .dcm分开
    for name in names:
        index = name.rfind('.')
        name = name[:index]
        c.append(name)

    for files in c:
        picture_path = "/home/shiya.xu/papers/models/datasets/rsna-pneumonia-detection-challenge/images/test/" + files + ".dcm" #dcm所在文件夹路径
        out_path = "/home/shiya.xu/papers/models/datasets/rsna-pneumonia-detection-challenge/images/val" + files + ".jpg" #要输出的jpg文件所在文件夹
        ds = pydicom.read_file(picture_path)
        img = ds.pixel_array  # 提取图像信息
        imageio.imsave(out_path, img)

    print('all is changed')

Dcm2jpg('/home/shiya.xu/papers/models/datasets/rsna-pneumonia-detection-challenge/images/test/') #传参，把dcm的文件夹路径传过来


