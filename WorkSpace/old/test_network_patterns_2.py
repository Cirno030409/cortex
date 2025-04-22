"""
実験：一つのニューロンへの再帰的な入力でのメモリー効果
"""

from brian2 import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
import Brian2_Framework.Plotters as Plotters
import matplotlib.pyplot as plt

params = {
    "I_noise"       : 0,
    "tauge"         : 5*ms,
    "taugi"         : 30*ms,
    "taum"          : 10*ms,
    "tautheta"      : 1e7*ms,
    "v_rev_e"       : 0,
    "v_rev_i"       : -100,
    "theta_dt"      : 0,
}

neuron = Conductance_LIF(params)
synapse = NonSTDP()
plotter = Plotters.Common_Plotter()

N = {}
S = {}

spikemon = {}
statemon = {}

N["1"] = neuron(1, 'exc', "N_1")
S["1"] = synapse(N["1"], N["1"], "exc", "S_1")

spikemon["1"] = SpikeMonitor(N["1"])
statemon["1"] = StateMonitor(N["1"], ["v", "I_noise"], record=True)

network = Network(N, S, spikemon, statemon)

neuron.change_params(N["1"], {"I_noise": 50})
network.run(100*ms)
neuron.change_params(N["1"], {"I_noise": 0})
network.run(100*ms)

plt.figure()
plotter.raster_plot(spikemon["1"], 2, 1, network.t, "Neuron 1")

plt.figure()
plotter.state_plot(statemon["1"], 0, "I_noise", 2, 1, network.t)

plt.show()





