from brian2 import *

start_scope()

eqs = '''
dv/dt = ((v_rest - v) + I) / tau : 1
v_rest : 1
I : 1
tau : second
'''

neuron_group1 = NeuronGroup(N=1, model=eqs, threshold="v>-30", reset="v=-65", method="euler")

neuron_group1.v = -65
neuron_group1.v_rest = -65
neuron_group1.tau = 10 * ms

neuron_group1.I = 50

statemon1 = StateMonitor(neuron_group1, "v", record=True)

run(100 * ms)

plot(statemon1.t/ms, statemon1.v[0])
legend(["Neuron group1[0]"])
xlabel("Time (ms)")
ylabel("Voltage (mV)")
show()

