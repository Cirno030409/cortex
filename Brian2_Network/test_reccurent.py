from brian2 import *
from Network.Neurons import *
from Network.Synapses import *
import Network.Plotters as Plotter

Neuron = Conductance_LIF()
Synapse = NonSTDP()

N = {}
S = {}

N_1 = Neuron(1, 'exc', 'N')
N_2 = Neuron(1, 'exc', 'N')
S = Synapse(N_1, N_2, 'S')




