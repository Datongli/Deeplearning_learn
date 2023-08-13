# -*- coding: utf-8 -*-

from pyswmm import Simulation, Subcatchments

with Simulation(r'C:\Users\ldt20\Desktop\swmm\220629-zzmx.inp') as sim:
    S1 = Subcatchments(sim)["S-29"]

    for step in sim:
        print(sim.current_time)
        print(S1.area)
        # print(type(S1.runoff))