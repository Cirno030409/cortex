from brian2 import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *

neuron_params = load_parameters("Brian2_Framework/parameters/Neuron/Conductance_LIF.json")


neuron1 = Conductance_LIF_Neuron(N=1, params=neuron_params)
neuron1.change_params({"I_noise": 100})
neuron2 = Conductance_LIF_Neuron(N=1, params=neuron_params)

synapse = Normal_Synapse(pre_neurons=neuron1, post_neurons=neuron2, params={"w": 1}, exc_or_inh="exc", name="synapse")

statemon = StateMonitor(neuron2, "v", record=True)

network = Network(neuron1, neuron2, synapse, statemon)

network.run(100*ms)

plot(statemon.t/ms, statemon.v[0])
show()
