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

import Brian2_Framework.Datasets as Datasets
import Brian2_Framework.Plotters as Plotters
import Brian2_Framework.Tools as tools
from Brian2_Framework.Networks import *
from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Validator import Validator


seed = 2
np.random.seed(seed)

# ===================================== 記録用パラメータ ==========================================
test_comment = "Center-Surround抑制を行うk=1nu=highepoch=2" #! 実験用コメント
PARAMS_PATH = "Brian2_Framework/parameters/WTA_CS/high_nu/WTA_CS_learn.json"
PLOT = False # プロットするか
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
# ===================================================================================================

params = tools.load_parameters(PARAMS_PATH) # パラメータを読み込み
name_test = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + test_comment
TARGET_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ
os.makedirs(os.path.join(TARGET_PATH, "LEARNING", "learning weight matrix"), exist_ok=True)
SAVE_PATH = TARGET_PATH
tools.save_parameters(SAVE_PATH, params) # パラメータをメモる

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成 
model = Center_Surround_WTA(PLOT, params_json_path=PARAMS_PATH) # ネットワークを作成

#! ===================================== シミュレーション ==========================================
print("[PROCESS] Running simulation...")
print(f"[INFO] Examination comment: {test_comment}")
all_labels = [] # 全Epochで入力された全ラベル
for j in tqdm(range(params["epoch"]), desc="epoch progress", dynamic_ncols=True): # エポック数繰り返す
    images, labels = Datasets.get_mnist_sample_equality_labels(params["n_samples"], "train") # テスト用の画像とラベルを取得
    all_labels.extend(labels)
    try:
        for i in tqdm(range(params["n_samples"]), desc="simulating", dynamic_ncols=True): # 画像枚数繰り返す
            if SAVE_WEIGHT_CHANGE_GIF: # 画像を記録
                if i % params["record_interval"] == 0:
                    
                    if i != 0:
                        plotter.firing_rate_heatmap(model.network["spikemon_for_assign"], 
                                                    params["exposure_time"]*(i-params["record_interval"]), 
                                                    params["exposure_time"]*i, 
                                                    save_fig=True, save_path=SAVE_PATH + "LEARNING/learning weight matrix/", 
                                                    n_this_fig=i+(j*params["n_samples"]))
                    plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], save_fig=True, save_path=SAVE_PATH + "LEARNING/learning weight matrix/", n_this_fig=i+(j*params["n_samples"]))
            tools.normalize_weight(model.network["S_0"], params["n_inp"] // 10, params["n_inp"], params["n_e"]) # 重みの正規化
            model.change_image(images[i], params["spontaneous_rate"]) # 入力画像の変更
            model.network.run(params["exposure_time"])
            tools.reset_network(model.network)
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")

# ===================================== ラベルの割り当て ==========================================
assigned_labels = tools.assign_labels2neurons(model.network["spikemon_for_assign"],params["n_e"], 10, all_labels, params["exposure_time"], 0*ms) # ニューロンにラベルを割り当てる
tools.memo_assigned_labels(SAVE_PATH, assigned_labels) # メモ
tools.save_assigned_labels(SAVE_PATH, assigned_labels) # 保存
print(f"[INFO] Saved assigned labels to {SAVE_PATH + 'assigned_labels.pkl'}")
weights = model.network["S_0"].w
np.save(SAVE_PATH + "weights.npy", weights) # 重みを保存(numpy)
if SAVE_WEIGHT_CHANGE_GIF:
    tools.make_gif(25, SAVE_PATH + "LEARNING/learning weight matrix", SAVE_PATH, "weight_change.gif") # GIFを保存
    
plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], title="weight plot of S0", save_fig=True, save_path=SAVE_PATH, n_this_fig="final_weight_plot", assigned_labels=assigned_labels)

# ===================================== シミュレーション結果のプロット ==========================================
if PLOT:
    plotter.set_simu_time(model.network.t)
    print("[PROCESS] Plotting results...")
    plotter.raster_plot([model.network["spikemon_0"], model.network["spikemon_1"], model.network["spikemon_2"]], fig_title="Raster plot of N0, N1, N2")
    plt.savefig(SAVE_PATH + "LEARNING/raster_plot_N0_N1_N2.svg")

    plotter.state_plot(model.network["statemon_1"], 0, ["v", "Ie", "Ii", "ge", "gi"], fig_title="State plot of N1")
    plt.savefig(SAVE_PATH + "LEARNING/state_plot_N1.svg")
    plt.show()

time.sleep(1)
SAVE_PATH = tools.change_dir_name(SAVE_PATH, "_comp/") # 完了したのでディレクトリ名を変更

