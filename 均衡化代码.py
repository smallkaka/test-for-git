# -*-coding:utf-8-*-
"""
File Name: image_histogram_operation.py
Program IDE: PyCharm
Date: 19:58
Create File By Author: Hong
"""
import cv2 as cv


def hist_equalization(image_path: str):
    # 直方图均衡化
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cv.imshow('input', img)
    result = cv.equalizeHist(img)
    cv.imshow('result', result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/daiyutong.png'
    hist_equalization(path)