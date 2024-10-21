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
test_comment = "Cortexテスト" #! 実験用コメント
PARAMS_PATH = "Brian2_Framework/parameters/Cortex/Cortex_learn.json" #! 使用するパラメータ
PARAMS_VALIDATE_PATH = "Brian2_Framework/parameters/Cortex/Cortex_validate.json" #! 使用する検証用パラメータ
PLOT = True # プロットするか
VALIDATION = False # Accuracyを計算するか
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
# ===================================================================================================

params = tools.load_parameters(PARAMS_PATH) # パラメータを読み込み
name_test = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + test_comment
TARGET_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ
os.makedirs(os.path.join(TARGET_PATH, "LEARNING", "learning weight matrix"), exist_ok=True)
SAVE_PATH = TARGET_PATH
tools.save_parameters(os.path.join(SAVE_PATH, "parameters.json"), params) # パラメータをメモる

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成
model = Cortex(PLOT, params_json_path=PARAMS_PATH) # ネットワークを作成

#! ===================================== シミュレーション ==========================================
print("[PROCESS] Running simulation...")
print(f"[INFO] Examination comment: {test_comment}")
all_labels = [] # 全Epochで入力された全ラベル
for j in tqdm(range(params["epoch"]), desc="epoch progress", dynamic_ncols=True): # エポック数繰り返す
    images, labels = Datasets.get_mnist_sample_equality_labels(params["n_samples"], "train") # テスト用の画像とラベルを取得
    all_labels.extend(labels)
    try:
        for i in tqdm(range(params["n_samples"]), desc="simulating", dynamic_ncols=True): # 画像枚数繰り返す
            # tools.normalize_weight(model.network["S_0"], params["n_inp"] // 10, params["n_inp"], params["n_e"]) # 重みの正規化
            model.change_image(images[i], params["spontaneous_rate"]) # 入力画像の変更
            model.network.run(params["exposure_time"]) # シミュレーション実行
            tools.reset_network(model.network) # ネットワークをリセット
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")

# ===================================== ラベルの割り当て ==========================================

# ===================================== シミュレーション結果のプロット ==========================================
if PLOT:
    plotter.set_simu_time(model.network.t) # シミュレーション時間を設定
    print("[PROCESS] Plotting results...")
    plotter.raster_plot([model.network["mc0_spikemon_N_inp"], model.network["mc0_spikemon_N_1"], model.network["mc1_spikemon_N_1"]], time_end=500, fig_title="Raster plot", save_path=SAVE_PATH + "LEARNING/Raster plot.png")

    plt.show()

time.sleep(1)
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

