from brian2 import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *
from Brian2_Framework.Plotters import *

defaultclock.dt = 0.5*ms

neuron_params = load_parameters("Brian2_Framework/parameters/Neuron/Conductance_LIF.json")


neuron1 = Conductance_LIF_Neuron(N=1, params=neuron_params, name="neuron1")
neuron1.I_noise = 800*pA
neuron2 = Conductance_LIF_Neuron(N=1, params=neuron_params, name="neuron2")

synapse = Normal_Synapse(pre_neurons=neuron1, post_neurons=neuron2, params={"w": 100*nS}, exc_or_inh="exc", name="synapse")

statemon = StateMonitor(neuron2, "v", record=True, name="statemon_neuron2")
spikemon = SpikeMonitor(neuron2, record=True, name="spikemon_neuron2")

network = Network(neuron1, neuron2, synapse, statemon, spikemon)

network.run(100*ms)

visualize_network(network)
state_plot(statemon, neuron_num=0, variable_names=["v"], time_end=100*ms)
raster_plot(spikemon, time_end=100*ms)
show()
