"""
実験：複数入力から同一出力を連想するネットワークの検証
"""
from brian2 import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
import Brian2_Framework.Plotters as Plotters
import matplotlib.pyplot as plt
import Brian2_Framework.Datasets as Datasets

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

Datasets.load_mnist()
images, labels = Datasets.get_mnist_image(label=1, n_samples=20, down_sample=2, dataset='train')

N = []
S = []

spikemon = {}
statemon = {}

layer_num = 10
for i in range(layer_num):
    N.append(neuron(5, 'exc', f"N_{i}"))
    
for i in range(layer_num):
    if i != layer_num - 1:
        S.append(synapse(N[i], N[i+1], "exc", f"S_{i}", connect="i==j"))

    spikemon[i] = SpikeMonitor(N[i])
    statemon[i] = StateMonitor(N[i], ["v", "I_noise"], record=True)

network = Network(N, S, spikemon, statemon)

print("I'm plotting...")
for i in range(5):
    if i != 0:
        N[0].I_noise[i-1] = 0
    N[0].I_noise[i] = 50
    network.run(16*ms)

plt.figure()
for i in range(layer_num):
    plotter.raster_plot(spikemon[i], layer_num, i+1, network.t, f"Neuron {i}")

plt.show()






