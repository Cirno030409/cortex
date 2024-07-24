import matplotlib.pyplot as plt
from brian2 import *

start_scope()

eqs = """
dv/dt = ((v_rest - v) + I) / tau : 1
v_rest : 1
I : 1
tau : second
"""

neuron_group1 = NeuronGroup(
    N=3, model=eqs, threshold="v>-30", reset="v=30; v=v_rest", method="euler"
)

neuron_group2 = NeuronGroup(
    N=3, model=eqs, threshold="v>-30", reset="v=30; v=v_rest", method="euler"
)

neuron_group1.v = -65
neuron_group2.v = -65
neuron_group1.v_rest = -65
neuron_group2.v_rest = -65
neuron_group1.tau = 10 * ms
neuron_group2.tau = 50 * ms

neuron_group1.I = 50

synapse = Synapses(neuron_group1, neuron_group2, on_pre="I += 2")

synapse.connect("i==j")

statemon1 = StateMonitor(neuron_group1, "v", record=True)
statemon2 = StateMonitor(neuron_group2, "v", record=True)

run(100 * ms)

plt.plot(statemon1.t / ms, statemon1.v[0])
plt.plot(statemon2.t / ms, statemon2.v[0])
plt.legend(["Neuron group1[0]", "Neuron group2[0]"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.show()
