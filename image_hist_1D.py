# -*-coding:utf-8-*-
"""
File Name: image_histogram_operation.py
Program IDE: Vscode
Date: 19:58
Create File By Author: Hong
在vscode里面画图 , 先将mabox打开 , 同时在vscode终端里面输入export DISPLAY=localhost:10.0
export DISPLAY=localhost:10.0
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def image_hist(image_path: str):
    """
    图像直方图是反映一个图像像素分布的统计表，其
    横坐标 : 代表了图像像素的种类，可以是灰度的，也可以是彩色的。
    纵坐标 : 代表了每一种颜色值在图像中的像素总数或者占所有像素个数的百分比。

    图像是由像素构成，那么反映像素分布的直方图往往可以作为图像一个很重要的特征。
    直方图的显示方式是左暗又亮，左边用于描述图像的暗度，右边用于描述图像的亮度。
    :param image_path: 传入查找像素的图像文件
    :return: 无返回值
    """
    # 一维直方图（单通道直方图）

    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)
    color = ('blue', 'green', 'red')

    # 使用plt内置函数直接绘制
    plt.hist(img.ravel(), 20, [0, 256])
    plt.show()

    # 一维像素直方图，也即是单通道直方图
    for i, color in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        print(hist)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = '/home/shiya.xu/Api/test_images/2.png'
    image_hist(path)