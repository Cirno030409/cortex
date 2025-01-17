from brian2 import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *
from Brian2_Framework.Networks import *
from Brian2_Framework.Plotters import *

params = load_parameters("Brian2_Framework/parameters/Mini_column_biological/learn.json")

network = Mini_Column_biological_3inh(params)

network.network["L4_N_pyr"].namespace["I_noise"] = 100

network.run(500*ms, report="text")

raster_plot(network.network["spikemon_L4_N_pyr"], time_end=500*ms)
show()

