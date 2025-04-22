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

seed = 2
np.random.seed(seed)

PARAMS_PATH = "Brian2_Framework/parameters/Chunk_WTA/Chunk_WTA_learn.json"
params = tools.load_parameters(PARAMS_PATH)

#! Parameters for recording
test_comment = "チャンクWTAテスト" #! Comment for the experiment
name_test = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + test_comment
PLOT = True # プロットするか
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
RECORD_INTERVAL = 50 # 記録する間隔
SAVE_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ

os.makedirs(SAVE_PATH) # 保存用ディレクトリを作成
print(f"[INFO] Created directory: {SAVE_PATH}")

# パラメータをメモる
tools.save_parameters(SAVE_PATH, params)

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成 
# ネットワークを作成
model = Chunk_WTA(PLOT, PARAMS_PATH)

#! Run simulation =====================================================================
print("[PROCESS] Running simulation...")
print(f"[INFO] Examination comment: {test_comment}")
for j in tqdm(range(params["epoch"]), desc="Epoch progress", dynamic_ncols=True): # エポック数繰り返す
    images, labels = Datasets.get_mnist_sample_equality_labels(params["n_samples"], "train") # テスト用の画像とラベルを取得
    chunks = []
    for i in range(params["n_samples"]):
        chunks.extend(Datasets.divide_image_into_chunks(images[i], params["chunk_size"]))
    try:
        for i in tqdm(range(len(chunks)), desc="Simulation progress", dynamic_ncols=True): # 画像枚数繰り返す
            if SAVE_WEIGHT_CHANGE_GIF: # 画像を記録
                if i % RECORD_INTERVAL == 0:
                    if i != 0:
                        plotter.firing_rate_heatmap(model.network["spikemon_1_exc"], 
                                                    params["exposure_time"]*(i-RECORD_INTERVAL), 
                                                    params["exposure_time"]*i, 
                                                    save_fig=True, save_path=SAVE_PATH, 
                                                    n_this_fig=str(i+(j*len(chunks)))+"_1")
                        plotter.firing_rate_heatmap(model.network["spikemon_2_exc"], 
                                                    params["exposure_time"]*(i-RECORD_INTERVAL), 
                                                    params["exposure_time"]*i, 
                                                    save_fig=True, save_path=SAVE_PATH, 
                                                    n_this_fig=str(i+(j*len(chunks)))+"_2")
                    plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], save_fig=True, save_path=SAVE_PATH, n_this_fig=str(i+(j*len(chunks)))+"_S0")
                    plotter.weight_plot(model.network["S_1_2"], n_pre=params["n_e"], n_post=params["n_e"], save_fig=True, save_path=SAVE_PATH, n_this_fig=str(i+(j*len(chunks)))+"_S12")
            tools.normalize_weight(model.network["S_0"], params["n_inp"]//10, params["n_inp"], params["n_e"]) # 重みの正規化
            tools.normalize_weight(model.network["S_1_2"], params["n_e"]//10, params["n_e"], params["n_e"]) # 重みの正規化
            model.set_input_image(chunks[i], params["spontaneous_rate"]) # 入力画像の設定
            model.run(params["exposure_time"])
            model.reset()
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")
#! =====================================================================================
print("[PROCESS] Assigning labels to neurons...")
assigned_labels = tools.assign_labels2neurons(model.network["spikemon_2_exc"], params["n_e"], 10, labels, params["exposure_time"], 0*ms) # ニューロンにラベルを割り当てる
tools.memo_assigned_labels(SAVE_PATH, assigned_labels) # メモ
tools.save_assigned_labels(SAVE_PATH, assigned_labels) # 保存
weights = model.network["S_0"].w
np.save(SAVE_PATH + "weights.npy", weights) # 重みを保存(numpy)
# GIFを保存
if SAVE_WEIGHT_CHANGE_GIF:
    print("[PROCESS] Saving weight change GIF...")
    tools.make_gif(25, SAVE_PATH, SAVE_PATH, "weight_change.gif")
    
plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], title="weight plot of S0")
plt.savefig(SAVE_PATH + "weight_plot_S0.png")

# シミュレーション結果をプロット
if PLOT:
    plotter.set_simu_time(model.network.t)
    print("[PROCESS] Plotting results...")
    plotter.raster_plot([model.network["spikemon_input"], model.network["spikemon_1_exc"], model.network["spikemon_2_exc"]], fig_title="Raster plot of N_input, N1_exc, N2_exc")
    plt.savefig(SAVE_PATH + f"raster_plot_N_input_N1_exc_N2_exc.png")
    
    plotter.state_plot(model.network["statemon_N_1_exc"], 0, ["v", "Ie", "Ii", "ge", "gi"], fig_title=f"State plot of N1_exc")
    plt.savefig(SAVE_PATH + f"state_plot_N1_exc.png")

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


