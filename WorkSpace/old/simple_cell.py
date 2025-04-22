import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from brian2.units import *
from Brian2_Framework.Networks import MiniColumn
from Brian2_Framework.Plotters import Plotter
from tqdm import tqdm

if __name__ == "__main__":
    
    # 5x5の画素値データ
    pixel_values = []
    pixel_values.append(np.array([
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0, 255, 255,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    ]))
    pixel_values.append(np.array([
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0, 255, 255, 255, 255, 255, 255, 255, 255,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0
    ]))

    # pixel_values.append(np.array([
    #     0,   0,   0,   0,   0,
    #     0,   0,   0, 255,   0,
    #     0,   0, 255,   0,   0,
    #     0, 255,   0,   0,   0,
    #     0,   0,   0,   0,   0
    # ]))
    # pixel_values.append(np.array([
    #     0,   0,   0,   0,   0,
    #     0, 255,   0,   0,   0,
    #     0,   0, 255,   0,   0,
    #     0,   0,   0, 255,   0,
    #     0,   0,   0,   0,   0
    # ]))

    # train_data, train_labels = mnist.load_data()


    max_rate = 50 # 最大発火率
    for i in range(len(pixel_values)):
        pixel_values[i] = pixel_values[i] / 255.0
        pixel_values[i] = pixel_values[i] * max_rate
        
    net = MiniColumn(n_inp=100, n_l4=100, n_l23=9, n_inh=9, monitors=True)
    
    print("[NOW LEARNING...]")
    EPOCH = 1
    for j in range(EPOCH):
        for i in tqdm(range(2)):
            net.run(inp=pixel_values[i].flatten() * Hz, duration=300 * ms)
        net.reset()

    print("[NOW PLOTTING... (Let's have a coffee. ^^)]")
    plotter = Plotter(net, column_id=0)
    plotter.draw_current()
    plotter.draw_potential()
    # plotter.draw_conductance()
    plotter.draw_raster_plot()
    plotter.draw_weight_changes(one_fig=True)
    plotter.draw_weight()
    # plotter.draw_firing_rate_changes()
    
    print("[ALL DONE.]")
    plt.show()
    