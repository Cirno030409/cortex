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

start_time = dt.now()

# ===================================== 記録用パラメータ ==========================================
PARAMS_PATH = "Brian2_Framework/parameters/Cortex/Cortex_learn.json" #! 使用するパラメータ
PARAMS_MC_PATH = "Brian2_Framework/parameters/Mini_Column/Mini_Column_learn.json" #! 使用するミニカラムのパラメータ
PARAMS_VALIDATE_PATH = "Brian2_Framework/parameters/Cortex/Cortex_validate.json" #! 使用するvalidation用パラメータ
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
# ===================================================================================================
params = tools.load_parameters(PARAMS_PATH) # パラメータを読み込み
params_mc = tools.load_parameters(PARAMS_MC_PATH) # ミニカラムのパラメータを読み込み
test_comment = f"Cortex - mc={params['n_mini_column']} labels={params['labels']} n={params['n_samples']}" #! 実験用コメント

np.random.seed(params["seed"]) # 乱数のシードを設定
name_test = dt.now().strftime("%Y%m%d%H%M%S_") + test_comment
TARGET_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ
for i in range(params["n_mini_column"]):
    os.makedirs(os.path.join(TARGET_PATH, "LEARNING", "learning weight matrix", f"mini column{i}"), exist_ok=True)
SAVE_PATH = TARGET_PATH
tools.save_parameters(os.path.join(SAVE_PATH, "learning_parameters.json"), params) # パラメータをメモる
tools.save_parameters(os.path.join(SAVE_PATH, "learning_mini_column_parameters.json"), params_mc) # ミニカラムのパラメータをメモる

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成
model = Cortex(params["enable_monitor"], params_cortex=params, params_mc=params_mc) # ネットワークを作成

#! ===================================== シミュレーション ==========================================
tools.print_simulation_start()
print(f"\n▶ Examination name: {name_test}\n")
all_labels = [] # 全Epochで入力された全ラベル
for j in tqdm(range(params["epoch"]), desc="epoch progress", dynamic_ncols=True): # エポック数繰り返す
    images, labels = Datasets.get_mnist_sample_equality_labels(params["n_samples"], labels=params["labels"], dataset="train") # テスト用の画像とラベルを取得
    all_labels.extend(labels)
    try:
        for i in tqdm(range(params["n_samples"]), desc="simulating", dynamic_ncols=True): # 画像枚数繰り返す
            if SAVE_WEIGHT_CHANGE_GIF: # 画像を記録
                if (i % params["record_interval"] == 0) or (i == 0) or (i == params["n_samples"]-1):
                    for k in range(params["n_mini_column"]): # ミニカラムごとに繰り返す
                        # ニューロンの発火率のヒートマップをプロット
                        plotter.firing_rate_heatmap(model.network[f"mc{k}_spikemon_for_assign"], 
                                                    params["exposure_time"]*(i-params["record_interval"]), 
                                                    params["exposure_time"]*i, 
                                                    save_fig=True, save_path=SAVE_PATH + f"LEARNING/learning weight matrix/mini column{k}/", 
                                                    n_this_fig=i+(j*params["n_samples"]))
                        # シナプスの重みのプロット
                        plotter.weight_plot(model.network[f"mc{k}_S_0"], n_pre=params_mc["n_inp"], n_post=params_mc["n_e"], save_path=SAVE_PATH + f"LEARNING/learning weight matrix/mini column{k}/", n_this_fig=i+(j*params["n_samples"]))
            for k in range(params["n_mini_column"]): # ミニカラムごとに繰り返す
                tools.normalize_weight(model.network[f"mc{k}_S_0"], params_mc["n_inp"] // 10, method="sum") # 重みの正規化
            model.set_input_image(images[i], params["spontaneous_rate"]) # 入力画像の設定
            if i < params["sync_weight_term"]: # 初期重み同期
                for k in range(params["n_mini_column"]):
                    model.network[f"mc{k}_S_1"].w[:] = 0 # 重みを0に設定
            else:
                for k in range(params["n_mini_column"]):
                    model.network[f"mc{k}_S_1"].w[:] = params_mc["static_synapse_params_ei"]["w"]
                    # model.network[f"mc{k}_N_1"].namespace["theta_dt"] = 0 # ホメスタシスの係数を0に設定
            model.run(params["exposure_time"])
            model.reset()
    except KeyboardInterrupt:
        print("[INFO] Simulation was interrupted by user.")
            
    # ===================================== ラベルの割り当て & 重みの保存 ==========================================
    for i in range(params["n_mini_column"]):
        weight = model.network[f"mc{i}_S_1"].w
        np.save(os.path.join(SAVE_PATH, f"mc{i}_weights.npy"), weight)
        assigned_labels = tools.assign_labels2neurons(model.network[f"mc{i}_spikemon_for_assign"], params_mc["n_e"], len(params["labels"]), all_labels, params["exposure_time"], 0*ms) # ニューロンにラベルを割り当てる
        tools.memo_assigned_labels(os.path.join(SAVE_PATH, f"mc{i}_assigned_labels.txt"), assigned_labels) # メモ
        tools.save_assigned_labels(os.path.join(SAVE_PATH, f"mc{i}_assigned_labels.pkl"), assigned_labels) # 保存


    # ===================================== シミュレーション結果のプロット ==========================================
    with open(SAVE_PATH + "LEARNING/input image labels.json", "w") as f:
        json.dump([int(label) for label in all_labels], f)  # numpy.int32をintに変換
    if SAVE_WEIGHT_CHANGE_GIF:
        for k in range(params["n_mini_column"]):
            plotter.weight_plot(model.network[f"mc{k}_S_0"], n_pre=params_mc["n_inp"], n_post=params_mc["n_e"], save_path=SAVE_PATH + f"LEARNING/", n_this_fig=f"weight_plot(mc{k})")
            tools.make_gif(25, SAVE_PATH + f"LEARNING/learning weight matrix/mini column{k}/", SAVE_PATH + "LEARNING/", f"weight_change(mc{k}).gif")
   
    tools.save_all_monitors(SAVE_PATH + f"LEARNING/Monitors/", model.network) # モニターを保存
    
    elapsed_time = dt.now() - start_time
    print(f"[INFO] Simulation time: {elapsed_time}")
    with open(os.path.join(SAVE_PATH, "LEARNING", "simulation_time.txt"), "w") as f:
        f.write(f"{elapsed_time}")
        
    time.sleep(3)
    try:
        SAVE_PATH = tools.change_dir_name(SAVE_PATH, "_comp/") # 完了したのでディレクトリ名を変更
    except:
        pass


