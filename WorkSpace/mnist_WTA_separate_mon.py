import io
import os
import pickle as pkl
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


seed = 3
np.random.seed(seed)


# ===================================== パラメータ ==========================================
test_comment = "WTA - 分割モニター保存テスト" #! 実験用コメント
PARAMS_PATH = "Brian2_Framework/parameters/WTA/membrane_tau_50/WTA_learn.json" #! 使用するパラメータ
MON_SEP_PARAMS_PATH = "Brian2_Framework/parameters/save_monitors_separately.json" #! モニター分割保存用パラメータ
PARAMS_VALIDATE_PATH = "Brian2_Framework/parameters/WTA/membrane_tau_50/WTA_validate.json" #! 使用する検証用パラメータ
SAVE_MONITORS = True # モニターを保存するか
SAVE_MONITORS_SEPARATELY = True # モニターを分割して保存するか
VALIDATION = False # Accuracyを計算するか
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
# ===================================================================================================

params = tools.load_parameters(PARAMS_PATH) # パラメータを読み込み
mon_sep_params = tools.load_parameters(MON_SEP_PARAMS_PATH) # モニター分割保存用パラメータを読み込み
name_test = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + test_comment
TARGET_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ
os.makedirs(os.path.join(TARGET_PATH, "LEARNING", "learning weight matrix"), exist_ok=True)
os.makedirs(os.path.join(TARGET_PATH, "LEARNING", "monitors"), exist_ok=True)
os.makedirs(os.path.join(TARGET_PATH, "LEARNING", "monitors", "separated_monitors"), exist_ok=True)
SAVE_PATH = TARGET_PATH
tools.save_parameters(os.path.join(SAVE_PATH, "parameters.json"), params) # パラメータをメモる

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成 
model = Diehl_and_Cook_WTA(SAVE_MONITORS, params) # ネットワークを作成

#! ===================================== シミュレーション ==========================================
print("[PROCESS] Running simulation...")
print(f"[INFO] Examination comment: {name_test}")
all_labels = [] # 全Epochで入力された全ラベル
for j in tqdm(range(params["epoch"]), desc="epoch progress", dynamic_ncols=True): # エポック数繰り返す
    images, labels = Datasets.get_mnist_sample_equality_labels(params["n_samples"], "train") # テスト用の画像とラベルを取得
    all_labels.extend(labels)
    try:
        for i in tqdm(range(params["n_samples"]), desc="simulating", dynamic_ncols=True): # 画像枚数繰り返す
            if SAVE_MONITORS_SEPARATELY and (i % mon_sep_params["separate_interval"] == 0):
                # モニターの保存、クリア、更新を一括で行う
                tools.save_all_monitors(
                    os.path.join(SAVE_PATH, "LEARNING", "monitors", "separated_monitors"), 
                    model.network, 
                    i//mon_sep_params["separate_interval"]
                )
            if SAVE_WEIGHT_CHANGE_GIF and (i % params["record_interval"] == 0 or i == params["n_samples"] - 1):
                plotter.firing_rate_heatmap(model.network["spikemon_for_assign"], 
                                            params["exposure_time"]*(i-params["record_interval"]), 
                                            params["exposure_time"]*i, 
                                            save_fig=True, save_path=SAVE_PATH + "LEARNING/learning weight matrix/", 
                                            n_this_fig=i+(j*params["n_samples"]))
                plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], save_fig=True, save_path=SAVE_PATH + "LEARNING/learning weight matrix/", n_this_fig=i+(j*params["n_samples"]))
            tools.normalize_weight(model.network["S_0"], params["n_inp"] // 10, params["n_inp"], params["n_e"]) # 重みの正規化
            model.set_input_image(images[i], params["spontaneous_rate"]) # 入力画像の設定
            model.run(params["exposure_time"])
            model.reset() # ネットワークをリセット
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")
tools.save_all_monitors(
    os.path.join(SAVE_PATH, "LEARNING", "monitors", "separated_monitors"), 
    model.network, 
    params["n_samples"]//mon_sep_params["separate_interval"]
)
merged_spikemon_for_assign = tools.merge_separated_monitors(os.path.join(SAVE_PATH, "LEARNING", "monitors", "separated_monitors", "spikemon_for_assign"))
tools.save_monitor(merged_spikemon_for_assign, os.path.join(SAVE_PATH, "LEARNING", "monitors", "spikemon_for_assign.pkl"))
# ===================================== ラベルの割り当て ==========================================
print("[PROCESS] Assigning labels to neurons...")
assigned_labels = tools.assign_labels2neurons(merged_spikemon_for_assign,params["n_e"], 10, all_labels, params["exposure_time"], 0*ms) # ニューロンにラベルを割り当てる
tools.memo_assigned_labels(SAVE_PATH, assigned_labels) # メモ
tools.save_assigned_labels(SAVE_PATH, assigned_labels) # 保存
weights = model.network["S_0"].w
np.save(SAVE_PATH + "weights.npy", weights) # 重みを保存(numpy)
if SAVE_WEIGHT_CHANGE_GIF:
    print("[PROCESS] Saving weight change GIF...")
    tools.make_gif(25, SAVE_PATH + "LEARNING/learning weight matrix/", SAVE_PATH, "weight_change.gif") # GIFを保存
    
plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], title="weight plot of S0", save_fig=True, save_path=SAVE_PATH, n_this_fig="final_weight_plot", assigned_labels=assigned_labels)

# ===================================== シミュレーション結果のプロット ==========================================
if SAVE_MONITORS:
    plotter.set_simu_time(model.network.t)
    print("[PROCESS] Plotting results...")
    plotter.raster_plot([model.network["spikemon_inp"], model.network["spikemon_N_1"], model.network["spikemon_N_2"]], time_end=300)
    plt.savefig(SAVE_PATH + "LEARNING/raster_plot_N0_N1_N2.png")

    plotter.state_plot(model.network["statemon_N_1"], 0, ["v", "Ie", "Ii", "ge", "gi"], time_end=300)
    plt.savefig(SAVE_PATH + "LEARNING/state_plot_N1.png")
    
    # モニターを保存
    tools.save_monitor(model.network["spikemon_inp"], os.path.join(SAVE_PATH, "LEARNING", "monitors", "spikemon_inp.pkl"))
    tools.save_monitor(model.network["spikemon_N_1"], os.path.join(SAVE_PATH, "LEARNING", "monitors", "spikemon_N_1.pkl"))
    tools.save_monitor(model.network["spikemon_N_2"], os.path.join(SAVE_PATH, "LEARNING", "monitors", "spikemon_N_2.pkl"))
    tools.save_monitor(model.network["statemon_N_1"], os.path.join(SAVE_PATH, "LEARNING", "monitors", "statemon_N_1.pkl"), ["v", "Ie", "Ii", "ge", "gi"])
    tools.save_monitor(model.network["statemon_N_2"], os.path.join(SAVE_PATH, "LEARNING", "monitors", "statemon_N_2.pkl"), ["v", "Ie", "Ii", "ge", "gi"])

time.sleep(3)
SAVE_PATH = tools.change_dir_name(SAVE_PATH, "_comp/") # 完了したのでディレクトリ名を変更

# ===================================== 精度の計算 ==========================================
if VALIDATION:
    params = tools.load_parameters(PARAMS_VALIDATE_PATH)
    validator = Validator(
                        target_path=SAVE_PATH, 
                        assigned_labels_path=f"{SAVE_PATH}/assigned_labels.pkl", 
                        params_json_path=PARAMS_VALIDATE_PATH,
                        network_type=params["network_type"])
    validator.validate(n_samples=params["n_samples"])
    


