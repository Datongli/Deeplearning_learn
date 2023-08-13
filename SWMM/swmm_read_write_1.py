"""
此文件专用于inp文件的读写学习与测试
"""
import pyswmm
import torch
from tqdm import tqdm
import pandas as pd
from pyswmm import Simulation, Nodes, Subcatchments, LidControl


# # 获取GPU设备
# if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
#     device = torch.device('cuda:0')
#     print('GPU')
# else:
#     device = torch.device('cpu')
#     print('CPU')

# inp_file:用于存放inp文件的地址
inp_flie = r'D:\PythonProject\Image_recognition\SWMM\220629-zzmx.inp'

# 打开一个inp文件
with Simulation(inp_flie) as sim:
    # 打印几个看看 i
    i = 0
    # sim.to(device)
    lid = LidControl(sim, False, 'A1')
    print(lid.surface)
    subcatchments = Subcatchments(sim)
    print(len(subcatchments))
    S1 = subcatchments['S-9']
    print(S1.area)
    # 更改汇水分区中管道的面积
    S1.area = 50
    print(S1.area)
    # # 读取每一个集水区
    for subcatchment in tqdm(Subcatchments(sim)):
        # 打印每一个集水区的名称
        print(subcatchment.subcatchmentid)
        # 打印每一个集水区的面积
        # 后续就可以在这里更改
        print(Subcatchments(sim)[subcatchment.subcatchmentid].area)
        if i <= 10:
            i += 1
        else:
            break

    sim.start()
    sim.step_advance(60)
    for step in tqdm(sim):
        print(S1.runoff)


