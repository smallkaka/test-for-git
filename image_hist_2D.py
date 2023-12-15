# -*-coding:utf-8-*-
"""
File Name: image_histogram_operation.py
Program IDE: vscode
Date: 19:58
Create File By Author: Hong

在vscode里面画图 , 先将mabox打开 , 同时在vscode终端里面输入export DISPLAY=localhost:10.0
export DISPLAY=localhost:10.0
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def image_hist2d(image_path: str):
    # 二维直方图
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('img', img)

    # 图像转HSV颜色空间
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [48, 48], [0, 180, 0, 256])
    dst = cv.resize(hist, (400, 400))

    # 像素归一化
    cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)

    # 色彩填充
    dst = cv.applyColorMap(np.uint8(dst), cv.COLORMAP_JET)

    cv.imshow('hist', dst)
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram')
    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = '/home/shiya.xu/Api/test_images/2.png'
    image_hist2d(path)