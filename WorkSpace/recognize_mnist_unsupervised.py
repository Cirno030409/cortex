import io
import os
import pickle as pkl
import pprint as p
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from tqdm import tqdm

import Brian2_Framework.Mnist as Mnist
import Brian2_Framework.Plotters as Plotters
import Brian2_Framework.Tools as tools
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from datetime import datetime

np.random.seed(1)

#! Neuron & Synapse Parameters
neuron_params_e = {
    "I_noise"       : 0,        # 定常入力電流
    "tauge"         : 1*ms,     # 興奮性ニューロンのコンダクタンスの時定数
    "taugi"         : 2*ms,     # 抑制性ニューロンのコンダクタンスの時定数
    "taum"          : 100*ms,    # 膜電位の時定数
    "theta_dt"      : 0.05,      # ホメオスタシスの発火閾値の上昇値
    "tautheta"      : 1e7*ms,   # ホメオスタシスの発火閾値の上昇値の減衰時定数
    "v_rev_e"       : 0,        # 興奮性ニューロンの平衡膜電位
    "v_rev_i"       : -100,     # 抑制性ニューロンの平衡膜電位
    "refractory"    : 2 * ms,   # 不応期
    "v_reset"       : -60,      # リセット電位
    "v_rest"        : -50,      # 静止膜電位
    "v_th"          : -40       # 発火閾値
}

neuron_params_i = {
    "I_noise"       : 0,        # 定常入力電流
    "tauge"         : 1*ms,     # 興奮性ニューロンのコンダクタンスの時定数
    "taugi"         : 2*ms,     # 抑制性ニューロンのコンダクタンスの時定数
    "taum"          : 100*ms,    # 膜電位の時定数
    "theta_dt"      : 0,      # ホメオスタシスの発火閾値の上昇値
    "tautheta"      : 1e7*ms,   # ホメオスタシスの発火閾値の上昇値の減衰時定数
    "v_rev_e"       : 0,        # 興奮性ニューロンの平衡膜電位
    "v_rev_i"       : -100,     # 抑制性ニューロンの平衡膜電位
    "refractory"    : 2 * ms,   # 不応期
    "v_reset"       : -60,      # リセット電位
    "v_rest"        : -50,      # 静止膜電位
    "v_th"          : -40       # 発火閾値
}

stdp_synapse_params = {
    "wmax": 1,              # 最大重み
    "wmin": 0,              # 最小重み
    "Apre": 1,           # 前ニューロンのスパイクトレースのリセット値
    "Apost": 1,             # 後ニューロンのスパイクトレースのリセット値
    "taupre": 20 * ms,      # 前ニューロンのスパイクトレースの時定数
    "taupost": 20 * ms,     # 後ニューロンのスパイクトレースの時定数
    "nu_pre": 0.0005,        # 学習率
    "nu_post": 0.000005,       # 学習率
    "alpha": 0,          # スパイクトレースの収束地点
    "sw": 1,             # 学習の有無の切り替え
}

static_synapse_params_ei = {
    "w": 30,
}

static_synapse_params_ie = {
    "w": 22,
}

# Make instances of neurons and synapses
neuron_e = Conductance_LIF(neuron_params_e)
neuron_i = Conductance_LIF(neuron_params_i)
neuron_inp = Poisson_Input()
synapse_ei = NonSTDP(static_synapse_params_ei)
synapse_ie = NonSTDP(static_synapse_params_ie)
synapse_stdp = STDP(stdp_synapse_params)

#! Network Parameters
train_or_test = "train"
n_samples = 40000 # 入力するMNISTデータの枚数
epoch = 2 # エポック数
n_inp = 784 # 入力層のニューロンの数
n_e = 100 # 興奮ニューロンの数
n_i = 100 # 抑制ニューロンの数
max_rate = 60 # 入力層の最大発火率
spontaneous_rate = 0 # 自発発火率 

# Parameters for recording
test_comment = "めっちゃいい感じ!!" #! Comment for the experiment
name_test = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PLOT = False
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
RECORD_INTERVAL = 50 # 記録する間隔
SAVE_PATH = "examined_data/" # 重みの変遷画像を保存するパス

os.makedirs(SAVE_PATH + name_test + "_" + test_comment)
print(f"[INFO] Created d irectory: {SAVE_PATH + name_test + '_' + test_comment}")
SAVE_PATH = SAVE_PATH + name_test + "_" + test_comment + "/"

# Note parameters to json
with open(SAVE_PATH + "parameters.json", "w") as f:
    parameters = {
        "n_samples": n_samples,
        "epoch": epoch,
        "n_inp": n_inp,
        "n_e": n_e,
        "n_i": n_i,
        "max_rate": max_rate,
        "spontaneous_rate": spontaneous_rate,
        "neuron_params_e": neuron_params_e,
        "neuron_params_i": neuron_params_i,
        "stdp_synapse_params": stdp_synapse_params,
        "static_synapse_params_ei": static_synapse_params_ei,
        "static_synapse_params_ie": static_synapse_params_ie
    }
    json.dump(parameters, f, indent=4, default=tools.convert_quantity)

N = [] # ニューロンリスト
S = [] # シナプスリスト

spikemon = {} # スパイクモニター
statemon_n = {} # ニューロンの状態モニター
statemon_s = {} # シナプスの状態モニター

# Create network
N.append(neuron_inp(n_inp, max_rate=max_rate))
N.append(neuron_e(n_e, "N_middle"))
N.append(neuron_i(n_i, "N_output"))

S.append(synapse_stdp(N[0], N[1], "S_0", connect=True)) # 入力層から興奮ニューロン
S.append(synapse_ei(N[1], N[2], "exc", "S_1", delay=0*ms, connect="i==j")) # 興奮ニューロンから抑制ニューロン
S.append(synapse_ie(N[2], N[1], "inh", "S_2", delay=0*ms, connect="i!=j")) # 側抑制
    
# Create monitors
for i, neuron in enumerate(N):
    spikemon[i] = SpikeMonitor(neuron, record=True)
    if PLOT:
        if i != 0:
            statemon_n[i] = StateMonitor(neuron, ["v", "I_noise", "Ie", "Ii", "ge", "gi"], record=50)
if PLOT:
    statemon_s= StateMonitor(S[0], ["w", "apre", "apost"], record=30000) 

network = Network(N, S, spikemon, statemon_n, statemon_s) # ネットワークを作成
plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成 

#! Run simulation =====================================================================
print("[PROCESS] Running simulation...")
print(f"[INFO] Simulation mode is <{train_or_test}>")
print(f"[INFO] Examination comment: {test_comment}")
assert train_or_test in ["train", "test"], f"[ERROR] train_or_test must be 'train' or 'test': {train_or_test}"
if train_or_test == "train":
    S[0].sw = 1 # 学習を行う
    SAVE_WEIGHT_CHANGE_GIF = False if train_or_test == "test" else SAVE_WEIGHT_CHANGE_GIF # テスト時は重みの変遷GIFを保存しない
elif train_or_test == "test":
    S[0].sw = 0 # 学習を行わない
    restore(filename=SAVE_PATH + "weights.b2", restore_random_state=True)
for j in range(epoch): # エポック数繰り返す
    images, labels = Mnist.get_mnist_sample_equality_labels(n_samples, train_or_test) # テスト用の画像とラベルを取得
    try:
        for i in tqdm(range(n_samples), desc="Simulation progress"):
            if SAVE_WEIGHT_CHANGE_GIF: # 画像を記録
                if i % RECORD_INTERVAL == 0:
                    plotter.weight_plot(S[0], n_pre=n_inp, n_post=n_e, save_fig=True, save_path=SAVE_PATH, n_this_fig=i+(j*n_samples))
            tools.normalize_weight(S[0], 78, n_inp, n_e) # 重みの正規化
            neuron_inp.change_image(tools.normalize_image_by_sum(images[i]), spontaneous_rate) # 入力画像の変更
            network.run(350*ms)
            neuron_inp.change_image(np.zeros((28, 28))) # 入力画像のリセット
            network.run(150*ms)
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")
#! =====================================================================================
# ニューロン
assigned_labels = tools.assign_labels2neurons(spikemon[1],n_e, 10, labels, 350*ms, 150*ms)
print(f"[INFO] Assigned labels: {assigned_labels}")
if train_or_test == "train":
    with open(SAVE_PATH + "assigned_labels.pkl", "wb") as f:
        pkl.dump(assigned_labels, f)
        print(f"[INFO] Saved assigned labels to {SAVE_PATH + 'assigned_labels.pkl'}")
    # 重みを保存
    store(filename=SAVE_PATH + "weights.b2")
    print(f"[INFO] Saved weights to {SAVE_PATH + 'weights.b2'}")
# GIFを保存
if SAVE_WEIGHT_CHANGE_GIF:
    print("[PROCESS] Saving weight change GIF...")
    tools.make_gif(25, SAVE_PATH, SAVE_PATH, "weight_change.gif")
    
plotter.weight_plot(S[0], n_pre=n_inp, n_post=n_e, title="weight plot of S0")
plt.savefig(SAVE_PATH + "weight_plot_S0.png")

# シミュレーション結果をプロット
if PLOT:
    plotter.set_simu_time(network.t)
    print("[PROCESS] Plotting results...")
    for i in range(3):
        plt.figure()
        plotter.raster_plot(spikemon[i], 1, 1, fig_title=f"Raster plot of N{i}")
        plt.savefig(SAVE_PATH + f"raster_plot_N{i}.png")

    plt.figure()
    plotter.state_plot(statemon_n[1], 0, "v", 5, 1, fig_title="State plot of N1")
    plotter.state_plot(statemon_n[1], 0, "Ie", 5, 2)
    plotter.state_plot(statemon_n[1], 0, "Ii", 5, 3)
    plotter.state_plot(statemon_n[1], 0, "ge", 5, 4)
    plotter.state_plot(statemon_n[1], 0, "gi", 5, 5)
    plt.savefig(SAVE_PATH + "state_plot_N1.png")

    plt.figure()
    plotter.state_plot(statemon_n[2], 0, "v", 5, 1)
    plotter.state_plot(statemon_n[2], 0, "Ie", 5, 2)
    plotter.state_plot(statemon_n[2], 0, "Ii", 5, 3)
    plotter.state_plot(statemon_n[2], 0, "ge", 5, 4)
    plotter.state_plot(statemon_n[2], 0, "gi", 5, 5)
    plt.savefig(SAVE_PATH + "state_plot_N2.png")

    plt.figure()
    plotter.state_plot(statemon_s, 0, "w", 3, 1)
    plotter.state_plot(statemon_s, 0, "apre", 3, 2)
    plotter.state_plot(statemon_s, 0, "apost", 3, 3)
    plt.savefig(SAVE_PATH + "state_plot_S0.png")

print("[PROCESS] Done.")

# 完了したのでディレクトリ名を変更
new_save_path = SAVE_PATH[:-1] + "_comp"
if not os.path.exists(new_save_path):
    os.makedirs(new_save_path)
for filename in os.listdir(SAVE_PATH):
    old_file = os.path.join(SAVE_PATH, filename)
    new_file = os.path.join(new_save_path, filename)
    os.rename(old_file, new_file)
time.sleep(1)
shutil.rmtree(SAVE_PATH)
SAVE_PATH = new_save_path + "/"
print(f"[INFO] ディレクトリ名を {new_save_path} に変更しました。")

# tools.print_firing_rate(spikemon[0])
plt.show()


