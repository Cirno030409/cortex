import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

b2.start_scope()
num_of_layer = 2
layer_hw = 2

layer_height, layer_width = 2, 2
num_of_neuron = layer_height * layer_width
neuron_groups = {}

i = 0
neuron_groups["layer0"] = b2.PoissonGroup(10, 100 * b2.Hz)

i = 1
neuron_groups["layer1"] = b2.PoissonGroup(10, 100 * b2.Hz)

# NetWork
net = b2.Network()
for neuron_group in neuron_groups.values():
    net.add(neuron_group)

M = b2.SpikeMonitor(neuron_groups[f"layer0"])

net.run(500 * b2.ms)

b2.plot(M.t / b2.ms, M.i, "|k")
b2.show()
