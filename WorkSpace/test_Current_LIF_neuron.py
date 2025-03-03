from brian2 import *

from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *

defaultclock.dt = 0.5*ms
duration = 30*ms

neuron_params = load_parameters("Brian2_Framework/parameters/Neuron/Conductance_LIF.json")
obj = {}

obj["neuron1"] = Current_LIF_Neuron(N=1, params=neuron_params, name="neuron1")
obj["neuron1"].I_noise = 1*nA
obj["neuron2"] = Current_LIF_Neuron(N=1, params=neuron_params, name="neuron2")
obj["synapse1_2"] = Normal_Synapse(obj["neuron1"], obj["neuron2"], name="synapse1_2", p=1, connect=True, params={"tau": 0.5*ms, "w": 100*pA, "delay": 1*ms}, exc_or_inh="exc")

obj["spikemon1"] = SpikeMonitor(obj["neuron1"], record=True, name="spikemon_neuron1")
obj["statemon1"] = StateMonitor(obj["neuron1"], ["v", "I_noise"], record=True, name="statemon_neuron1")

obj["spikemon2"] = SpikeMonitor(obj["neuron2"], record=True, name="spikemon_neuron2")
obj["statemon2"] = StateMonitor(obj["synapse1_2"], ["syn"], record=True, name="statemon_synapse1_2")


network = Network(obj.values())


network.run(duration)

visualize_network(network)
plot_all_monitors(network, time_end=duration)

show()
