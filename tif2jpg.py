# 批量tiff转jpg
# 代码中路径更改为自己图像存放路径即可
from PIL import Image
import os
 
imagesDirectory = "/home/shiya.xu/papers/models/datasets/chronme/images/train"  # tiff图片所在文件夹路径
distDirectory = "/home/shiya.xu/papers/models/datasets/chronme/images/train_1"# 要存放jpg格式的文件夹路径
for imageName in os.listdir(imagesDirectory):
    imagePath = os.path.join(imagesDirectory, imageName)
    image = Image.open(imagePath)# 打开tiff图像
    distImagePath = os.path.join(distDirectory, imageName[:-4]+'.jpg')# 更改图像后缀为.jpg，并保证与原图像同名
    print(imagePath)
    image.save(distImagePath)# 保存jpg图像