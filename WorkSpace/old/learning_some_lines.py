import matplotlib.pyplot as plt
import numpy as np
from brian2.units import *
from keras.datasets import mnist
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
        
    net = MiniColumn(model="Izhikevich", n_inp=100, n_l4=100, n_l23=9, n_inh=9, monitors=True)
    
    for idx, pixel_value in enumerate(pixel_values):
        plt.figure()
        plt.imshow(pixel_value.reshape(10, 10), cmap="gray")
        plt.title(f"Pixel Value Image {idx+1}")
        # plt.axis("off")
        for i in range(11):
            plt.axhline(i - 0.5, color='white', linewidth=0.5)
            plt.axvline(i - 0.5, color='white', linewidth=0.5)
        plt.axis("off")
    plt.show()
    
    print("[NOW LEARNING...]")
    EPOCH = 1
    for j in range(EPOCH):
        for i in tqdm(range(len(pixel_values))):
            net.run(inp=pixel_values[i].flatten(), max_rate=50 * Hz, duration=500 * ms)
        net.reset()

    print("[NOW PLOTTING... (Let's have a coffee. ^^)]")
    plotter = Plotter(net, column_id=0)
    plotter.draw_current()
    plotter.draw_potential()
    plotter.draw_threshold_changes()
    # plotter.draw_conductance()
    plotter.draw_raster_plot()
    # plotter.draw_weight_changes(one_fig=True)
    plotter.draw_weight()
    # plotter.draw_firing_rate_changes()
    
    print("[ALL DONE.]")
    plt.show()
    