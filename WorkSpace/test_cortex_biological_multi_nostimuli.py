from brian2 import *
from tqdm import tqdm

from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *
import sys

seeds = [0]

# コマンドライン引数からシード値を取得
if len(sys.argv) > 1:
    try:
        seeds = [int(sys.argv[1])]
        print(f"コマンドライン引数からシード値を設定しました: {seeds[0]}")
    except ValueError:
        print(f"警告: 引数 '{sys.argv[1]}' を整数に変換できません。デフォルトのシード値を使用します。")

for seed in seeds:
    np.random.seed(seed)
    defaultclock.dt = 0.5*ms
    duration = 500*ms


    params = load_parameters("Brian2_Framework/parameters/Cortex_Microcircuit_multiple/params.json")
    params_mc = load_parameters(params["micro_circuit_params_path"])

    save_dir = os.path.join("data", "cortex_biological", "multiple", "without stimulate", f"n={params['n_micro_circuit']}", f"seed={seed}")
    os.makedirs(save_dir, exist_ok=True)

    # run network
    net = Jung_H_Lee_Cortex_MicroCircuit_multiple(params)
    visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))

    net.run(duration, report="text")
    
    # save plots
    os.makedirs(os.path.join(save_dir, "popmon"), exist_ok=True)
    plot_all_monitors(net.network, time_end=duration, monitor_type=["popmon"], save_dir_path=os.path.join(save_dir, "popmon"))
    os.makedirs(os.path.join(save_dir, "spikemon"), exist_ok=True)
    plot_all_monitors(net.network, time_end=duration, monitor_type=["spikemon"], save_dir_path=os.path.join(save_dir, "spikemon"))
    os.makedirs(os.path.join(save_dir, "statemon"), exist_ok=True)
    plot_all_monitors(net.network, time_end=duration, monitor_type=["statemon"], save_dir_path=os.path.join(save_dir, "statemon"))
    os.makedirs(os.path.join(save_dir, "monitors"), exist_ok=True)
    save_all_monitors(net.network, os.path.join(save_dir, "monitors"))
    
    save_parameters(os.path.join(save_dir, "parameters.json"), params)
    save_parameters(os.path.join(save_dir, "parameters_mc.json"), params_mc)
    plt.close('all')
# show()