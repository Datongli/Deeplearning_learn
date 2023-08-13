import pyswmm
from tqdm import tqdm
import pandas as pd
from pyswmm import simulation, links, nodes, Subcatchments
from pyswmm import Simulation, Nodes
import time


node = nodes
link = links

# 设置inp文件路径
inp_file = r'D:\PythonProject\Image_recognition\SWMM\1121-2D.inp'
# inp_file = r"C:\Users\ldt20\Desktop\swmm\220629-zzmx.inp"


"""
里面可以加载的是：
Link, Links, LidControls, LidGroups, Node, Nodes, Subcatchment(集水区), Subcatchments, Simulation,
SystemStats, RainGages, RainGage, Output
"""

# 创建模拟对象并运行模拟
with pyswmm.Simulation(inp_file) as sim:
    subcatch_object = Subcatchments(sim)
    SC1 = subcatch_object["SJ1"]
    print(SC1.area)
    SC1.area = 50
    sim.start()
    sim.step_advance(60)  # 超参量，表示每步仿真时间间隔为600秒
    results = []

    print(" ")
    print(SC1.area)

    for step in tqdm(sim):
    # for step in sim:
        print(SC1.runoff)
        # 获取节点和链接的属性值
        # node_results = {node.nodeid: node.depth for node in nodes.Nodes(sim)}
        # link_results = {link.linkid: link.flowrate for link in link.Links(sim)}
        # 将结果存储为一个字典
        # row = {**node_results, **link_results}
        # results.append(row)
    # print(results)
# 将结果转换为 DataFrame，并保存为 csv 文件
# df = pd.DataFrame(results)
# df.to_csv('./output.csv', index=False)
print("over")

