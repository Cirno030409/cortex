from brian2 import *
from tqdm import tqdm

from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Plotters import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *
import sys

seeds = [0, 1, 2, 3, 4]

# コマンドライン引数からシード値を取得
duration_arg = None
n_mc = None
if len(sys.argv) > 1:
    try:
        seeds = [int(sys.argv[1])]
        duration_arg = int(sys.argv[2]) * ms
        n_mc = int(sys.argv[3])
        print(f"コマンドライン引数からシード値を設定しました: {seeds[0]}")
        print(f"コマンドライン引数からdurationを設定しました: {duration_arg}")
        print(f"コマンドライン引数からn_mcを設定しました: {n_mc}")
    except ValueError:
        raise ValueError(f"警告: 引数エラーが発生しました。")

for seed in tqdm(seeds, desc="trying seeds"):
    np.random.seed(seed)
    defaultclock.dt = 0.5*ms
    duration = 500*ms
    if duration_arg is not None:
        duration = duration_arg


    params = load_parameters("Brian2_Framework/parameters/Cortex_Microcircuit_multiple/params.json")
    params_mc = load_parameters(params["micro_circuit_params_path"])
    
    if n_mc is not None:
        params["n_micro_circuit"] = n_mc

    save_dir = os.path.join("data", "cortex_biological", "multiple", "without stimulate", f"n={params['n_micro_circuit']}", f"seed={seed}")
    os.makedirs(save_dir, exist_ok=True)

    # run network
    net = Jung_H_Lee_Cortex_MicroCircuit_multiple(params)
    visualize_network(net.network, save_path=os.path.join(save_dir, "Network Structure.png"))

    net.run(duration, report="text")
    
    # save plots
    plot_all_monitors(net.network, time_end=duration, monitor_type=["statemon"], save_dir_path=save_dir)
    
    os.makedirs(os.path.join(save_dir, "monitors"), exist_ok=True)
    save_all_monitors(net.network, os.path.join(save_dir, "monitors"))
    
    save_parameters(os.path.join(save_dir, "parameters.json"), params)
    save_parameters(os.path.join(save_dir, "parameters_mc.json"), params_mc)
    plt.close('all')
# show()