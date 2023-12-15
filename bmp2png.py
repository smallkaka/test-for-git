import os
import time
from PIL import Image


def Bmp2Png(file_path):
    for fileName in os.listdir(file_path):
        if fileName.split('.')[1] in ['bmp', 'BMP']:
            new_fileName = fileName.split('.')[0] + '.png'
            im = Image.open(fileName)
            im.save(new_fileName)
            print("Finish convert")

def main():
    start_time = time.perf_counter()
    file_path = "/home/shiya.xu/papers/DAG4MIA/code/Data/REFUGE/Non-Glaucoma/label1"
    try:
        Bmp2Png(file_path)
    except:
        print("It's not a img file")
    end_time = time.perf_counter()
    print("PIL costs %f s" % (end_time - start_time))

if __name__ == '__main__':
    main()