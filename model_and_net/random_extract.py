import os
import random
import shutil

"""
从文件夹里面随机挑选出一定数量的文件的程序
"""


def moveFile(fileDir, trainDir, number):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    # rate1 = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
    rate1 = number/filenumber
    picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)
    for name in sample1:
        file_path = os.path.join(fileDir, name)
        new_file_path = os.path.join(trainDir, name)
        shutil.copyfile(file_path, new_file_path)


if __name__ == '__main__':
    fileDir = r"D:\学习\大创\data\训练数据集\data\Track1+CoughVid 谱图合集\Track1+CoughVid logMel\positive"
    trainDir = r"C:\Users\ldt20\Desktop\small\positive"

    moveFile(fileDir, trainDir, 500)
    print("finish!!")
