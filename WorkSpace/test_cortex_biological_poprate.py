from brian2 import *
from tqdm import tqdm

from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *

defaultclock.dt = 0.5*ms
duration = 1000*ms

params = load_parameters("Brian2_Framework/parameters/Cortex_MicroCircuit/learn.json")

net = Jung_H_Lee_Cortex_MicroCircuit(params)
visualize_network(net.network)
# plot_all_monitors(net.network, time_end=duration, smooth_window=10)

input_rates = np.arange(0, 100, 10) * Hz

rates_L23_pyr = []
rates_L23_sst = []
rates_L23_vip = []
rates_L4_exc = []
rates_L4_inh = []
rates_L5_exc = []
rates_L5_inh = []
rates_L6_exc = []
rates_L6_inh = []
rates_N_inp = []
for i, input_rate in tqdm(enumerate(input_rates), desc="input_rate progress", dynamic_ncols=True, total=len(input_rates)):
    net.set_input_neuron_rate(input_rate)
    net.run(duration, report="text")
    print(f"======> input_rate: {input_rate} Hz")
    print("N_inpの発火率:", get_population_rate(net.network["0_spikemon_N_inp"], start_time=duration*i, end_time=duration*(i+1)))
    print("L4_N_excの発火率:", get_population_rate(net.network["0_spikemon_L4_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
    print("L23_N_pyrの発火率:", get_population_rate(net.network["0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
    print("L23_N_sstの発火率:", get_population_rate(net.network["0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
    print("L23_N_vipの発火率:", get_population_rate(net.network["0_spikemon_L23_N_vip"], start_time=duration*i, end_time=duration*(i+1)))
    print("L4_N_inhの発火率:", get_population_rate(net.network["0_spikemon_L4_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
    print("L5_N_excの発火率:", get_population_rate(net.network["0_spikemon_L5_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
    print("L5_N_inhの発火率:", get_population_rate(net.network["0_spikemon_L5_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
    print("L6_N_excの発火率:", get_population_rate(net.network["0_spikemon_L6_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
    print("L6_N_inhの発火率:", get_population_rate(net.network["0_spikemon_L6_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L4_exc.append(get_population_rate(net.network["0_spikemon_L4_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L23_pyr.append(get_population_rate(net.network["0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L23_sst.append(get_population_rate(net.network["0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L23_vip.append(get_population_rate(net.network["0_spikemon_L23_N_vip"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L4_inh.append(get_population_rate(net.network["0_spikemon_L4_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L5_exc.append(get_population_rate(net.network["0_spikemon_L5_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L5_inh.append(get_population_rate(net.network["0_spikemon_L5_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L6_exc.append(get_population_rate(net.network["0_spikemon_L6_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
    rates_L6_inh.append(get_population_rate(net.network["0_spikemon_L6_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
    rates_N_inp.append(get_population_rate(net.network["0_spikemon_N_inp"], start_time=duration*i, end_time=duration*(i+1)))

figure()
title("Firing rate of each layer while increasing input rate")
plot(input_rates, rates_L23_pyr, label="L23_N_pyr")
plot(input_rates, rates_L23_sst, label="L23_N_sst")
plot(input_rates, rates_L23_vip, label="L23_N_vip")
plot(input_rates, rates_L4_exc, label="L4_N_exc")
plot(input_rates, rates_L4_inh, label="L4_N_inh")
plot(input_rates, rates_L5_exc, label="L5_N_exc")
plot(input_rates, rates_L5_inh, label="L5_N_inh")
plot(input_rates, rates_L6_exc, label="L6_N_exc")
plot(input_rates, rates_L6_inh, label="L6_N_inh")

xlabel("Neuron spike rate of input Poisson neuron (Hz)")
ylabel("Firing rate of each layer (Hz)")
legend()
show()