import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from brian2.units import *
from Brian2_Framework.Networks import MiniColumn
from Brian2_Framework.Plotters import Plotter
from tqdm import tqdm

if __name__ == "__main__":
    pixel_values = []
    pixel_values.append(np.array([
    0
]))
        
    net_Izhi = MiniColumn(model="Izhikevich", n_inp=1, n_l4=1, n_l23=1, n_inh=0, monitors=True)
    net_LIF = MiniColumn(model="LIF", n_inp=1, n_l4=1, n_l23=1, n_inh=0, monitors=True)
    print("[NOW LEARNING...]")
    for i in tqdm(range(len(pixel_values))):
        net_Izhi.run(current_inp=20, max_rate=20 * Hz, duration=500 * ms)
        net_LIF.run(current_inp=20, max_rate=80 * Hz, duration=500 * ms)
    
    print("[NOW PLOTTING... (Let's have a coffee. ^^)]")
    plotter_Izhi = Plotter(net_Izhi, column_id=0)
    plotter_Izhi.draw_current()
    plotter_Izhi.draw_potential()
    plotter_Izhi.draw_raster_plot()
    
    plotter_LIF = Plotter(net_LIF, column_id=1)
    plotter_LIF.draw_current()
    plotter_LIF.draw_potential()
    plotter_LIF.draw_raster_plot()
    
    print("[ALL DONE.]")
    plt.show()
    