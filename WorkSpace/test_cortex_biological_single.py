"""
ニューロンタイプごとの入力に対する発火特性をプロットする
"""
from brian2 import *
from tqdm import tqdm
import warnings
import os
import mpld3

# 警告を包括的に抑制
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # すべてのRuntimeWarningを抑制
warnings.filterwarnings("ignore", message="Mean of empty slice")  # 空のスライスの平均値警告を抑制
warnings.filterwarnings("ignore", message="invalid value encountered in divide")  # 不正な値での除算警告を抑制
warnings.filterwarnings("ignore", category=UserWarning)  # UserWarningも抑制

from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *
import sys

seeds = [0, 1, 2, 3]

for seed in seeds:
    np.random.seed(seed)
    defaultclock.dt = 0.5*ms
    step = 10
    duration = 500*ms
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "Times New Roman"

    save_dir = "data/cortex_biological/single"

    params = load_parameters("Brian2_Framework/parameters/Cortex_MicroCircuit/params.json") # マイクロ回路モデルのパラメータ

    net = Jung_H_Lee_Cortex_MicroCircuit(params) # マイクロ回路モデル
    visualize_network(net.network) # ネットワークの可視化

    #! =======================================
    simulate_mode = sys.argv[1] if len(sys.argv) > 1 else "all_zero"
    # all : 局所回路にポアソン入力を与え，ニューロンタイプごとの発火率をプロット
    # all_zero : 入力なしで自発発火のみ
    # pyr-sst : sstニューロンにポアソン入力を与えたときのpyrニューロンの発火率をプロット
    # sst-sst : sstニューロンに定常電流を与えたときのsstニューロンの発火率をプロット
    # pyr-vip : vipニューロンにポアソン入力を与えたときのpyrニューロンの発火率をプロット
    # sst-vip : vipニューロンに定常電流を与えたときのsstニューロンの発火率をプロット
    #! =======================================

    print(f"simulate_mode: {simulate_mode}")

    if simulate_mode == "all" or simulate_mode == "all_zero":
        input_rates = np.arange(0, 210, step) * Hz
        if simulate_mode == "all_zero":
            input_rates = [0] * Hz
            save_dir = os.path.join(save_dir, "spontaneous firing", f"seed={seed}")
            duration = 1000*ms
        else:
            save_dir = os.path.join(save_dir, "Firing rate of each cell while increasing circuit input", f"seed={seed}")

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
            print("N_inpの発火率:", get_population_rate(net.network["M0_spikemon_N_inp"], start_time=duration*i, end_time=duration*(i+1)))
            print("L4_N_excの発火率:", get_population_rate(net.network["M0_spikemon_L4_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            print("L23_N_pyrの発火率:", get_population_rate(net.network["M0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
            print("L23_N_sstの発火率:", get_population_rate(net.network["M0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
            print("L23_N_vipの発火率:", get_population_rate(net.network["M0_spikemon_L23_N_vip"], start_time=duration*i, end_time=duration*(i+1)))
            print("L4_N_inhの発火率:", get_population_rate(net.network["M0_spikemon_L4_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            print("L5_N_excの発火率:", get_population_rate(net.network["M0_spikemon_L5_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            print("L5_N_inhの発火率:", get_population_rate(net.network["M0_spikemon_L5_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            print("L6_N_excの発火率:", get_population_rate(net.network["M0_spikemon_L6_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            print("L6_N_inhの発火率:", get_population_rate(net.network["M0_spikemon_L6_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L4_exc.append(get_population_rate(net.network["M0_spikemon_L4_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L23_pyr.append(get_population_rate(net.network["M0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L23_sst.append(get_population_rate(net.network["M0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L23_vip.append(get_population_rate(net.network["M0_spikemon_L23_N_vip"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L4_inh.append(get_population_rate(net.network["M0_spikemon_L4_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L5_exc.append(get_population_rate(net.network["M0_spikemon_L5_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L5_inh.append(get_population_rate(net.network["M0_spikemon_L5_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L6_exc.append(get_population_rate(net.network["M0_spikemon_L6_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L6_inh.append(get_population_rate(net.network["M0_spikemon_L6_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            rates_N_inp.append(get_population_rate(net.network["M0_spikemon_N_inp"], start_time=duration*i, end_time=duration*(i+1)))

        fig = figure(figsize=(12, 10))
        title("Firing rate of each cell while increasing circuit input")
        plot(input_rates/Hz, rates_L23_pyr/Hz, label="L23_N_pyr", marker=".")
        plot(input_rates/Hz, rates_L23_sst/Hz, label="L23_N_sst", marker=".")
        plot(input_rates/Hz, rates_L23_vip/Hz, label="L23_N_vip", marker=".")
        plot(input_rates/Hz, rates_L4_exc/Hz, label="L4_N_exc", marker=".")
        plot(input_rates/Hz, rates_L4_inh/Hz, label="L4_N_inh", marker=".")
        plot(input_rates/Hz, rates_L5_exc/Hz, label="L5_N_exc", marker=".")
        plot(input_rates/Hz, rates_L5_inh/Hz, label="L5_N_inh", marker=".")
        plot(input_rates/Hz, rates_L6_exc/Hz, label="L6_N_exc", marker=".")
        plot(input_rates/Hz, rates_L6_inh/Hz, label="L6_N_inh", marker=".")
        xlabel(f"Spike rate of input poisson neuron (Hz)")
        ylabel("Firing rate of each layer (Hz)")
        legend(loc="upper left", bbox_to_anchor=(1, 1))
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of each cell.png"))
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of each cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "pyr-sst":
        save_dir = os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_pyr = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            net.run(duration, report="text")
            rates_L23_pyr.append(get_population_rate(net.network["M0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of Pyr cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L23_pyr/Hz, marker=".", markersize=5, color="black")
        xlabel("Input current to SST cell (pA)")
        ylabel("Firing rate of Pyr cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to SST cell.png"))
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "sst-sst":
        save_dir = os.path.join(save_dir, "Firing rate of SST cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_sst = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            net.run(duration, report="text")
            rates_L23_sst.append(get_population_rate(net.network["M0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of SST cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L23_sst/Hz, marker=".", markersize=5, color="black")
        xlabel("Input current to SST cell (pA)")
        ylabel("Firing rate of SST cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of SST cell while increasing current input to SST cell.png"))
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of SST cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "pyr-vip":
        save_dir = os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to VIP cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_pyr = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_vip"].I_noise = current
            net.run(duration, report="text")
            rates_L23_pyr.append(get_population_rate(net.network["M0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of Pyr cell while increasing current input to VIP cell")
        plot(input_currents/pA, rates_L23_pyr/Hz, marker=".", markersize=5, color="black")
        xlabel("Input current to VIP cell (pA)")
        ylabel("Firing rate of Pyr cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to VIP cell.png"))
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to VIP cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "sst-vip":
        save_dir = os.path.join(save_dir, "Firing rate of SST cell while increasing current input to VIP cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_sst = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_vip"].I_noise = current
            net.run(duration, report="text")
            rates_L23_sst.append(get_population_rate(net.network["M0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of SST cell while increasing current input to VIP cell")
        plot(input_currents/pA, rates_L23_sst/Hz, marker=".", markersize=5, color="black")
        xlabel("Input current to VIP cell (pA)")
        ylabel("Firing rate of SST cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of SST cell while increasing current input to VIP cell.png"))
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of SST cell while increasing current input to VIP cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    else:
        raise ValueError(f"simulate_mode: {simulate_mode} is not supported")

    plt.close()