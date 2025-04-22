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

# GUIが使えない環境の場合はmatplotlibのバックエンドをnon-GUI modeに設定
import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    print("GUI環境が検出されなかったため、matplotlibのバックエンドをnon-GUI modeに設定しました")
    matplotlib.use("Agg")

from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *
import sys

seeds = [0, 1, 2, 3]

current_input_mode = True # True: 定常電流入力モード, False: ポアソン入力モード
I_pyr = 366*pA
I_pv = 362*pA
I_vip = 370*pA
I_sst = 361*pA

for seed in tqdm(seeds, desc="trying seeds", dynamic_ncols=True, total=len(seeds)):
    np.random.seed(seed)
    defaultclock.dt = 0.5*ms
    step = 10
    duration = 500*ms
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "Times New Roman"

    save_dir = "data/cortex_biological/single"

    params = load_parameters("Brian2_Framework/parameters/Cortex_Microcircuit/params.json") # マイクロ回路モデルのパラメータ
    if current_input_mode:
        params["ex_fibers"]["enable"] = False
    net = Jung_H_Lee_Cortex_MicroCircuit(params) # マイクロ回路モデル
    visualize_network(net.network) # ネットワークの可視化

    #! =======================================
    try:
        simulate_mode = sys.argv[1]
        print(f"==========> simulate_mode: {simulate_mode}")
    except:
        raise ValueError(f"specity simulate mode by command line argument")
    # all : 局所回路にポアソン入力を与え，ニューロンタイプごとの発火率をプロット
    # all_reverse : 局所回路にポアソン入力を降順で与え，ニューロンタイプごとの発火率をプロット
    # all_zero : 入力なしで自発発火のみ
    # pyr-sst : sstニューロンにポアソン入力を与えたときのpyrニューロンの発火率をプロット
    # exc-sst : sstニューロンにポアソン入力を与えたときのL4, L5, L6のexcニューロンの発火率をプロット
    # inh-sst : sstニューロンにポアソン入力を与えたときのL4, L5, L6のinhニューロンの発火率をプロット
    # vip-sst : sstニューロンにポアソン入力を与えたときのvipニューロンの発火率をプロット
    # pv-sst : sstニューロンにポアソン入力を与えたときのpvニューロンの発火率をプロット
    # sst-sst : sstニューロンに定常電流を与えたときのsstニューロンの発火率をプロット
    # pyr-vip : vipニューロンにポアソン入力を与えたときのpyrニューロンの発火率をプロット
    # sst-vip : vipニューロンに定常電流を与えたときのsstニューロンの発火率をプロット
    #! =======================================

    if simulate_mode == "all" or simulate_mode == "all_reverse" or simulate_mode == "all_zero":
        if simulate_mode == "all_reverse":
            input_rates = np.arange(210, 0, -step) * Hz
            save_dir = os.path.join(save_dir, "Firing rate of each cell while decreasing circuit input(reverse)", f"seed={seed}")
        elif simulate_mode == "all_zero":
            input_rates = [0] * Hz
            save_dir = os.path.join(save_dir, "spontaneous firing", f"seed={seed}")
            duration = 1000*ms
        else:
            input_rates = np.arange(0, 210, step) * Hz
            save_dir = os.path.join(save_dir, "Firing rate of each cell while increasing circuit input", f"seed={seed}")

        rates_L23_pyr = []
        rates_L23_pv = []
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
            try:
                rates_L4_exc.append(get_population_rate(net.network["M0_spikemon_L4_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L23_pyr.append(get_population_rate(net.network["M0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L23_pv.append(get_population_rate(net.network["M0_spikemon_L23_N_pv"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L23_sst.append(get_population_rate(net.network["M0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L23_vip.append(get_population_rate(net.network["M0_spikemon_L23_N_vip"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L4_inh.append(get_population_rate(net.network["M0_spikemon_L4_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L5_exc.append(get_population_rate(net.network["M0_spikemon_L5_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L5_inh.append(get_population_rate(net.network["M0_spikemon_L5_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L6_exc.append(get_population_rate(net.network["M0_spikemon_L6_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
                rates_L6_inh.append(get_population_rate(net.network["M0_spikemon_L6_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
                rates_N_inp.append(get_population_rate(net.network["M0_spikemon_N_inp"], start_time=duration*i, end_time=duration*(i+1)))
            except:
                print("monitor not found")

        fig = figure(figsize=(12, 10))
        title("Firing rate of each cell while increasing circuit input")
        try:
            plot(input_rates/Hz, rates_L23_pyr/Hz, label="L23_N_pyr", marker=".")
            plot(input_rates/Hz, rates_L23_sst/Hz, label="L23_N_sst", marker=".")
            plot(input_rates/Hz, rates_L23_vip/Hz, label="L23_N_vip", marker=".")
            plot(input_rates/Hz, rates_L23_pv/Hz, label="L23_N_pv", marker=".")
            plot(input_rates/Hz, rates_L4_exc/Hz, label="L4_N_exc", marker=".")
            plot(input_rates/Hz, rates_L4_inh/Hz, label="L4_N_inh", marker=".")
            plot(input_rates/Hz, rates_L5_exc/Hz, label="L5_N_exc", marker=".")
            plot(input_rates/Hz, rates_L5_inh/Hz, label="L5_N_inh", marker=".")
            plot(input_rates/Hz, rates_L6_exc/Hz, label="L6_N_exc", marker=".")
            plot(input_rates/Hz, rates_L6_inh/Hz, label="L6_N_inh", marker=".")
        except Exception as e:
            print(f"monitor not found: {e}")
        xlabel(f"Spike rate of poisson input neuron (Hz)")
        ylabel("Firing rate of each layer (Hz)")
        legend(loc="upper left", bbox_to_anchor=(1, 1))
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of each cell.png"), dpi=300, bbox_inches="tight")
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of each cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "pyr-sst":
        save_dir = os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_pyr = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            if current_input_mode:
                net.network["M0_L23_N_pyr"].I_noise = I_pyr
                net.network["M0_L23_N_vip"].I_noise = I_vip
                net.network["M0_L23_N_pv"].I_noise = I_pv
                save_dir = os.path.join(save_dir, "current input mode")
            net.run(duration, report="text")
            rates_L23_pyr.append(get_population_rate(net.network["M0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(12, 10))
        title("Firing rate of Pyr cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L23_pyr/Hz, marker=".", markersize=5, color="black")
        xlabel("Current input to SST cell (pA)")
        ylabel("Firing rate of Pyr cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to SST cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "exc-sst":
        save_dir = os.path.join(save_dir, "Firing rate of exc cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L4_exc = []
        rates_L5_exc = []
        rates_L6_exc = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            net.run(duration, report="text")
            rates_L4_exc.append(get_population_rate(net.network["M0_spikemon_L4_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L5_exc.append(get_population_rate(net.network["M0_spikemon_L5_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L6_exc.append(get_population_rate(net.network["M0_spikemon_L6_N_exc"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of exc cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L4_exc/Hz, marker=".", markersize=5, color="black")
        xlabel("Current input to SST cell (pA)")
        ylabel("Firing rate of exc cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of exc cell while increasing current input to SST cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of exc cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "inh-sst":
        save_dir = os.path.join(save_dir, "Firing rate of inh cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L4_inh = []
        rates_L5_inh = []
        rates_L6_inh = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            net.run(duration, report="text")
            rates_L4_inh.append(get_population_rate(net.network["M0_spikemon_L4_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L5_inh.append(get_population_rate(net.network["M0_spikemon_L5_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
            rates_L6_inh.append(get_population_rate(net.network["M0_spikemon_L6_N_inh"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of inh cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L4_inh/Hz, marker=".", markersize=5, color="black")
        xlabel("Current input to SST cell (pA)")
        ylabel("Firing rate of inh cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of inh cell while increasing current input to SST cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of inh cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "vip-sst":
        save_dir = os.path.join(save_dir, "Firing rate of vip cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_vip = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            net.run(duration, report="text")
            rates_L23_vip.append(get_population_rate(net.network["M0_spikemon_L23_N_vip"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of vip cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L23_vip/Hz, marker=".", markersize=5, color="black")
        xlabel("Current input to SST cell (pA)")
        ylabel("Firing rate of vip cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of vip cell while increasing current input to SST cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of vip cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "pv-sst":
        save_dir = os.path.join(save_dir, "Firing rate of pv cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_pv = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            net.run(duration, report="text")
            rates_L23_pv.append(get_population_rate(net.network["M0_spikemon_L23_N_pv"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of pv cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L23_pv/Hz, marker=".", markersize=5, color="black")
        xlabel("Current input to SST cell (pA)")
        ylabel("Firing rate of pv cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of pv cell while increasing current input to SST cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of pv cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "sst-sst":
        save_dir = os.path.join(save_dir, "Firing rate of SST cell while increasing current input to SST cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_sst = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_sst"].I_noise = current
            if current_input_mode:
                net.network["M0_L23_N_pyr"].I_noise = I_pyr
                net.network["M0_L23_N_vip"].I_noise = I_vip
                net.network["M0_L23_N_pv"].I_noise = I_pv
                save_dir = os.path.join(save_dir, "current input mode")
            net.run(duration, report="text")
            rates_L23_sst.append(get_population_rate(net.network["M0_spikemon_L23_N_sst"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of SST cell while increasing current input to SST cell")
        plot(input_currents/pA, rates_L23_sst/Hz, marker=".", markersize=5, color="black")
        xlabel("Current input to SST cell (pA)")
        ylabel("Firing rate of SST cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of SST cell while increasing current input to SST cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of SST cell while increasing current input to SST cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    elif simulate_mode == "pyr-vip":
        save_dir = os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to VIP cell", f"seed={seed}")
        input_currents = np.arange(0, 400, step) * pA
        rates_L23_pyr = []
        for i, current in tqdm(enumerate(input_currents), desc="input_current progress", dynamic_ncols=True, total=len(input_currents)):
            net.network["M0_L23_N_vip"].I_noise = current
            if current_input_mode:
                net.network["M0_L23_N_pyr"].I_noise = I_pyr
                net.network["M0_L23_N_sst"].I_noise = I_sst
                net.network["M0_L23_N_pv"].I_noise = I_pv
                save_dir = os.path.join(save_dir, "current input mode")
            net.run(duration, report="text")
            rates_L23_pyr.append(get_population_rate(net.network["M0_spikemon_L23_N_pyr"], start_time=duration*i, end_time=duration*(i+1)))
        fig = figure(figsize=(10, 10))
        title("Firing rate of Pyr cell while increasing current input to VIP cell")
        plot(input_currents/pA, rates_L23_pyr/Hz, marker=".", markersize=5, color="black")
        xlabel("Current input to VIP cell (pA)")
        ylabel("Firing rate of Pyr cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to VIP cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of Pyr cell while increasing current input to VIP cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
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
        xlabel("Current input to VIP cell (pA)")
        ylabel("Firing rate of SST cell (Hz)")
        os.makedirs(save_dir, exist_ok=True)
        savefig(os.path.join(save_dir, "Firing rate of SST cell while increasing current input to VIP cell.png"), dpi=300)
        mpld3.save_html(fig, os.path.join(save_dir, "Firing rate of SST cell while increasing current input to VIP cell.html"))
        plot_all_monitors(net.network, monitor_type=["all"], save_dir_path=save_dir, smooth_window=10)
        visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))
        save_all_monitors(net.network, save_path=os.path.join(save_dir, "monitors"))
        save_parameters(os.path.join(save_dir, "parameters.json"), params)
    else:
        raise ValueError(f"simulate_mode: {simulate_mode} is not supported")

    plt.close()