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
import pickle as pkl

seed = 2
np.random.seed(seed)
# ===================================== 記録用パラメータ ==========================================
test_comment = "Cortex - カラム間抑制結合(全結合)" #! 実験用コメント
PARAMS_PATH = "Brian2_Framework/parameters/Cortex/Cortex_learn.json" #! 使用するパラメータ
PARAMS_VALIDATE_PATH = "Brian2_Framework/parameters/Cortex/Cortex_validate.json" #! 使用するvalidation用パラメータ
PLOT = True # プロットするか
VALIDATION = False # Accuracyを計算するか
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
# ===================================================================================================

params = tools.load_parameters(PARAMS_PATH) # パラメータを読み込み
params_mc = tools.load_parameters(params["mini_column_params_path"]) # ミニカラムのパラメータを読み込み
name_test = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + test_comment
TARGET_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ
for i in range(params["n_mini_column"]):
    os.makedirs(os.path.join(TARGET_PATH, "LEARNING", "learning weight matrix", f"mini column{i}"), exist_ok=True)
SAVE_PATH = TARGET_PATH
tools.save_parameters(os.path.join(SAVE_PATH, "parameters.json"), params) # パラメータをメモる

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成
model = Cortex(enable_monitor=PLOT, params_json_path=PARAMS_PATH) # ネットワークを作成

#! ===================================== シミュレーション ==========================================
print("▶ Running simulation...")
print(f"▶ Examination name: {name_test}")
all_labels = [] # 全Epochで入力された全ラベル
for j in tqdm(range(params["epoch"]), desc="epoch progress", dynamic_ncols=True): # エポック数繰り返す
    images, labels = Datasets.get_mnist_sample_equality_labels(params["n_samples"], "train") # テスト用の画像とラベルを取得
    all_labels.extend(labels)
    try:
        for i in tqdm(range(params["n_samples"]), desc="simulating", dynamic_ncols=True): # 画像枚数繰り返す
            if SAVE_WEIGHT_CHANGE_GIF: # 画像を記録
                if (i % params["record_interval"] == 0) or (i == 0):
                    if i != 0:
                        for k in range(params["n_mini_column"]):
                            plotter.firing_rate_heatmap(model.network[f"mc{k}_spikemon_for_assign"], 
                                                        params["exposure_time"]*(i-params["record_interval"]), 
                                                        params["exposure_time"]*i, 
                                                        save_fig=True, save_path=SAVE_PATH + f"LEARNING/learning weight matrix/mini column{k}/", 
                                                        n_this_fig=i+(j*params["n_samples"]))
                            plotter.weight_plot(model.network[f"mc{k}_S_0"], n_pre=params_mc["n_inp"], n_post=params_mc["n_e"], save_path=SAVE_PATH + f"LEARNING/learning weight matrix/mini column{k}/", n_this_fig=i+(j*params["n_samples"]))
                            tools.normalize_weight(model.network[f"mc{k}_S_0"], params_mc["n_inp"] // 10, params_mc["n_inp"], params_mc["n_e"]) # 重みの正規化
            model.change_image(images[i], params["spontaneous_rate"]) # 入力画像の変更
            model.network.run(params["exposure_time"]) # シミュレーション実行
            tools.reset_network(model.network) # ネットワークをリセット
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")


# ===================================== シミュレーション結果のプロット ==========================================
if PLOT:
    plotter.set_simu_time(model.network.t) # シミュレーション時間を設定
    print("▶ Plotting results...")
    time_end = 10000
    
    for k in range(params["n_mini_column"]):
        plotter.raster_plot([model.network["spikemon_inp"], model.network[f"mc{k}_spikemon_for_assign"], model.network[f"mc{k}_spikemon_N_2"]], time_end=time_end, save_path=SAVE_PATH + f"LEARNING/Raster plot(mc{k}).png")
        plotter.state_plot(model.network[f"mc{k}_statemon_N_2"], 0, ["v", "ge", "gi"], time_end=time_end, save_path=SAVE_PATH + f"LEARNING/State plot(mc{k}).png")
    plt.show()

pkl.dump(model.network["mc0_spikemon_for_assign"], open(SAVE_PATH + "LEARNING/spikemon.pkl", "wb"))

time.sleep(1)
SAVE_PATH = tools.change_dir_name(SAVE_PATH, "_comp/") # 完了したのでディレクトリ名を変更


