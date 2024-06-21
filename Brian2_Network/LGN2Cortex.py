import matplotlib.pyplot as plt
import numpy as np
from brian2.units import *
from Network.Networks import MiniColumn
from Network.Plotters import Plotter
from tqdm import tqdm

if __name__ == "__main__":
    
    # 5x5の画素値データ
    pixel_values = []
    pixel_values.append(np.array([
        0,   0,  0,   0,   0,
        0,   0,   0,   0,   0,
        0, 255, 255, 255,   0,
        0,   0,   0,   0,   0,
        0,   0,   0,   0,   0
    ]))
    # pixel_values.append(np.array([
    #     0,   0,   0,   0,   0,
    #     0,   0, 255,   0,   0,
    #     0,   0, 255,   0,   0,
    #     0,   0, 255,   0,   0,
    #     0,   0,   0,   0,   0
    # ]))


    max_rate = 50 * Hz # 最大発火率
    for i in range(len(pixel_values)):
        pixel_values[i] = pixel_values[i] / 255.0
        pixel_values[i] = pixel_values[i] * max_rate
        
        
    net = MiniColumn(n_inp=25, n_l4=25, n_l23=9, n_inh=0, monitors=True)
    net.run(inp=pixel_values[0].flatten(), duration=400 * ms)
    
    plotter = Plotter(net, 0)
    plotter.draw_current()
    plotter.draw_potential()
    plotter.draw_raster_plot(400 *ms)
    plotter.draw_weight_changes(one_fig=True)
    plotter.draw_weight()
    plotter.draw_conductance(synapse_num=[0, 11, 12, 13])
    plt.show()
    