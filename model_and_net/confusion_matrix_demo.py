"""
浅浅绘制一下新的混淆矩阵，使用大一些的字体
"""
import imageio.v2 as imageio
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import itertools
import os
import random
# import cv2
import matplotlib.pyplot as plt
import numpy as np
# PaddyDataSet
from modeltools import plot_confusion_matrix


"""
浅浅绘制一下新的混淆矩阵，使用大一些的字体
"""
# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# 初始化混淆矩阵
cnf_matrix = np.zeros([2, 2])
cnf_matrix[0][0] = 2989
cnf_matrix[0][1] = 1478
cnf_matrix[1][0] = 1046
cnf_matrix[1][1] = 3447
Confusion_matrix_path = os.path.join(r"C:\Users\ldt20\Desktop\图片", "resneet18" + nowTime + ".svg")


classes = ['negative', 'positive']
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix', path=Confusion_matrix_path)
print('finish')