from brian2 import *

start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''
G = [0]*2
G[0] = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', method='exact')


M = StateMonitor(G[0], 'v', record=0)

run(50*ms)
plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v');

show()