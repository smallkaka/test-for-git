import pandas as pd
import os
import shutil

# 读取表格文件+填写你的csv文件的位置
f = open("/home/shiya.xu/Api/train.csv", "rb")
list = pd.read_csv(f)

# 进行分类----填写你要分类文件夹的标签,有多少就写多少
for i in ['cup','glass','fork','knife','plate','spoon','test']:
    if not os.path.exists(i):
        os.mkdir(i)
    listnew = list[list["type"] == i]#type是你csv文件里面的你要处理的那一列的列名称
    l = listnew["id"].tolist()#image这里是你的处理文件的名字的列名称
    j = str(i)
    for each in l:
        #这里是你数据文件放置的位置
        shutil.copy('/home/shiya.xu/Api/images/' + each, j)