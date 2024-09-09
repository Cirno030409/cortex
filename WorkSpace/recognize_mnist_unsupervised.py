import io
import os
import pickle as pkl
import pprint as p
import shutil
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from tqdm import tqdm

import Brian2_Framework.Mnist as Mnist
import Brian2_Framework.Plotters as Plotters
import Brian2_Framework.Tools as tools
from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *

seed = 2
np.random.seed(seed)

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
    "taum"          : 10*ms,    # 膜電位の時定数
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
    "nu_pre": 0.01,        # 学習率
    "nu_post": 0.0001,       # 学習率
    "alpha": 0,          # スパイクトレースの収束地点
    "sw": 1,             # 学習の有無の切り替え
}

static_synapse_params_ei = {
    "w": 30,
}

static_synapse_params_ie = {
    "w": 22,
}

# 読み込み用
WEIGHT_PATH = "examined_data/2024_09_04_11_23_11_めっちゃいい感じ!!_comp/weights.b2"
ASSIGNED_LABELS_PATH = "examined_data/2024_09_04_11_23_11_めっちゃいい感じ!!_comp/assigned_labels.pkl"

#! Network Parameters
n_samples = 40000 # 入力するMNISTデータの枚数
epoch = 2 # エポック数
n_inp = 784 # 入力層のニューロンの数
n_e = 100 # 興奮ニューロンの数
n_i = 100 # 抑制ニューロンの数
max_rate = 60 # 入力層の最大発火率
spontaneous_rate = 0 # 自発発火率

#! Parameters for recording
test_comment = "2Epoch回したら？" #! Comment for the experiment
name_test = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + test_comment
PLOT = False # プロットするか
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
RECORD_INTERVAL = 100 # 記録する間隔
SAVE_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ

os.makedirs(SAVE_PATH) # 保存用ディレクトリを作成
print(f"[INFO] Created directory: {SAVE_PATH}")

# パラメータをメモる
with open(SAVE_PATH + "parameters.json", "w") as f:
    parameters = {
        "n_samples": n_samples,
        "epoch": epoch,
        "seed": seed,
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

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成 
# ネットワークを作成
model = Diehl_and_Cook_WTA(PLOT,n_inp, n_e, n_i, max_rate, neuron_params_e, neuron_params_i, static_synapse_params_ei, static_synapse_params_ie, stdp_synapse_params)
    
#! Run simulation =====================================================================
print("[PROCESS] Running simulation...")
print(f"[INFO] Examination comment: {test_comment}")
for j in tqdm(range(epoch), desc="Epoch progress"): # エポック数繰り返す
    images, labels = Mnist.get_mnist_sample_equality_labels(n_samples, "train") # テスト用の画像とラベルを取得
    try:
        for i in tqdm(range(n_samples), desc="Simulation progress"): # 画像枚数繰り返す
            if SAVE_WEIGHT_CHANGE_GIF: # 画像を記録
                if i % RECORD_INTERVAL == 0:
                    plotter.weight_plot(model.network["S_0"], n_pre=n_inp, n_post=n_e, save_fig=True, save_path=SAVE_PATH, n_this_fig=i+(j*n_samples))
            tools.normalize_weight(model.network["S_0"], 78, n_inp, n_e) # 重みの正規化
            model.change_image(tools.normalize_image_by_sum(images[i]), spontaneous_rate) # 入力画像の変更
            model.network.run(350*ms)
            tools.reset_network(model.network)
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")
#! =====================================================================================
print("[PROCESS] Assigning labels to neurons...")
assigned_labels = tools.assign_labels2neurons(model.network["spikemon_1"],n_e, 10, labels, 350*ms, 0*ms) # ニューロンにラベルを割り当てる
print(f"[INFO] Assigned labels: ")
for i in range(len(assigned_labels)):
    print(f"\tneuron {i}: {assigned_labels[i]}")
with open(SAVE_PATH + "parameters.json", "a") as f:
    f.write(f"\nassigned_labels: {assigned_labels}")
with open(SAVE_PATH + "assigned_labels.pkl", "wb") as f:
    pkl.dump(assigned_labels, f)
    print(f"[INFO] Saved assigned labels to {SAVE_PATH + 'assigned_labels.pkl'}")
weights = model.network["S_0"].w
np.save(SAVE_PATH + "weights.npy", weights) # 重みを保存(numpy)
store(filename=SAVE_PATH + "weights.b2") # 重みを保存(Brian2)
print(f"[INFO] Saved weights to {SAVE_PATH + 'weights.b2'}")
# GIFを保存
if SAVE_WEIGHT_CHANGE_GIF:
    print("[PROCESS] Saving weight change GIF...")
    tools.make_gif(25, SAVE_PATH, SAVE_PATH, "weight_change.gif")
    
plotter.weight_plot(model.network["S_0"], n_pre=n_inp, n_post=n_e, title="weight plot of S0")
plt.savefig(SAVE_PATH + "weight_plot_S0.png")

# シミュレーション結果をプロット
if PLOT:
    plotter.set_simu_time(model.network.t)
    print("[PROCESS] Plotting results...")
    for i in range(3):
        plt.figure()
        plotter.raster_plot(model.network[f"spikemon_{i}"], 1, 1, fig_title=f"Raster plot of N{i}")
        plt.savefig(SAVE_PATH + f"raster_plot_N{i}.png")

    plt.figure()
    plotter.state_plot(model.network["statemon_1"], 0, "v", 5, 1, fig_title="State plot of N1")
    plotter.state_plot(model.network["statemon_1"], 0, "Ie", 5, 2)
    plotter.state_plot(model.network["statemon_1"], 0, "Ii", 5, 3)
    plotter.state_plot(model.network["statemon_1"], 0, "ge", 5, 4)
    plotter.state_plot(model.network["statemon_1"], 0, "gi", 5, 5)
    plt.savefig(SAVE_PATH + "state_plot_N1.png")

    plt.figure()
    plotter.state_plot(model.network["statemon_2"], 0, "v", 5, 1, fig_title="State plot of N2")
    plotter.state_plot(model.network["statemon_2"], 0, "Ie", 5, 2)
    plotter.state_plot(model.network["statemon_2"], 0, "Ii", 5, 3)
    plotter.state_plot(model.network["statemon_2"], 0, "ge", 5, 4)
    plotter.state_plot(model.network["statemon_2"], 0, "gi", 5, 5)
    plt.savefig(SAVE_PATH + "state_plot_N2.png")

    plt.figure()
    plotter.state_plot(model.network["statemon_S"], 0, "w", 3, 1, fig_title="State plot of S0")
    plotter.state_plot(model.network["statemon_S"], 0, "apre", 3, 2)
    plotter.state_plot(model.network["statemon_S"], 0, "apost", 3, 3)
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


