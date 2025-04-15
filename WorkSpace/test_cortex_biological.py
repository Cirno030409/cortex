from brian2 import *
from tqdm import tqdm

from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *

np.random.seed(0)
defaultclock.dt = 0.5*ms
duration = 200*ms

params = load_parameters("Brian2_Framework/parameters/Cortex_Microcircuit/params.json")

net = Jung_H_Lee_Cortex_MicroCircuit(params)
visualize_network(net.network)

# net.run(duration, report="text")

# plot_all_monitors(net.network, smooth_window=10)
# save_all_monitors(net.network, save_path="data/monitors")
show()