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
from Brian2_Framework.Validator import Validator

seed = 2

params = tools.load_parameters("Brian2_Framework/parameters/WTA.json")

# 読み込み用
WEIGHT_PATH = "examined_data/2024_09_04_11_23_11_めっちゃいい感じ!!_comp/weights.b2"
ASSIGNED_LABELS_PATH = "examined_data/2024_09_04_11_23_11_めっちゃいい感じ!!_comp/assigned_labels.pkl"

#! Parameters for recording
test_comment = "change dir test" #! Comment for the experiment
name_test = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + test_comment
PLOT = False # プロットするか
VALIDATION = True # Accuracyを計算するか
SAVE_WEIGHT_CHANGE_GIF = True # 重みの変遷.GIFを保存するか
RECORD_INTERVAL = 50 # 記録する間隔
SAVE_PATH = "examined_data/" + name_test + "/" # 色々保存するディレクトリ

os.makedirs(SAVE_PATH) # 保存用ディレクトリを作成
print(f"[INFO] Created directory: {SAVE_PATH}")

# パラメータをメモる
tools.save_parameters(SAVE_PATH, params)

plotter = Plotters.Common_Plotter() # プロットを行うインスタンスを作成 
# ネットワークを作成
model = Diehl_and_Cook_WTA(PLOT,
                           params["n_inp"], 
                           params["n_e"], 
                           params["n_i"], 
                           params["max_rate"], 
                           params["neuron_params_e"], 
                           params["neuron_params_i"], 
                           params["static_synapse_params_ei"], 
                           params["static_synapse_params_ie"], 
                           params["stdp_synapse_params"])

#! Run simulation =====================================================================
print("[PROCESS] Running simulation...")
print(f"[INFO] Examination comment: {test_comment}")
all_labels = [] # 全Epochで入力された全ラベル
for j in tqdm(range(params["epoch"]), desc="epoch progress", dynamic_ncols=True): # エポック数繰り返す
    images, labels = Mnist.get_mnist_sample_equality_labels(params["n_samples"], "train") # テスト用の画像とラベルを取得
    all_labels.extend(labels)
    try:
        for i in tqdm(range(params["n_samples"]), desc="simulating", dynamic_ncols=True): # 画像枚数繰り返す
            if SAVE_WEIGHT_CHANGE_GIF: # 画像を記録
                if i % RECORD_INTERVAL == 0:
                    if i != 0:
                        plotter.firing_rate_heatmap(model.network["spikemon_for_assign"], 
                                                    params["exposure_time"]*(i-RECORD_INTERVAL), 
                                                    params["exposure_time"]*i, 
                                                    save_fig=True, save_path=SAVE_PATH, 
                                                    n_this_fig=i+(j*params["n_samples"]))
                    plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], save_fig=True, save_path=SAVE_PATH, n_this_fig=i+(j*params["n_samples"]))
            tools.normalize_weight(model.network["S_0"], params["n_inp"] // 10, params["n_inp"], params["n_e"]) # 重みの正規化
            model.change_image(images[i], params["spontaneous_rate"]) # 入力画像の変更
            model.network.run(params["exposure_time"])
            tools.reset_network(model.network)
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.")
#! =====================================================================================
print("[PROCESS] Assigning labels to neurons...")
assigned_labels = tools.assign_labels2neurons(model.network["spikemon_for_assign"],n_e, 10, all_labels, exposure_time, 0*ms) # ニューロンにラベルを割り当てる
print(f"[INFO] Assigned labels: ")
for i in range(len(assigned_labels)):
    print(f"\tneuron {i}: {assigned_labels[i]}")
with open(SAVE_PATH + "assigned_labels.txt", "w") as f:
    f.write("[Assigned labels]\n")
    for i in range(len(assigned_labels)):
        f.write(f"\tneuron {i}: {assigned_labels[i]}\n")
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
    
plotter.weight_plot(model.network["S_0"], n_pre=params["n_inp"], n_post=params["n_e"], title="weight plot of S0", save_fig=True, save_path=SAVE_PATH, n_this_fig="final_weight_plot", assigned_labels=assigned_labels)

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
    plt.show()

print("[PROCESS] Done.")
time.sleep(1)
SAVE_PATH = tools.change_dir_name(SAVE_PATH, "_comp/") # 完了したのでディレクトリ名を変更
# Accuracyを計算
if VALIDATION:
    validator = Validator(
                        weight_path=f"{SAVE_PATH}/weights.npy", 
                        assigned_labels_path=f"{SAVE_PATH}/assigned_labels.pkl", 
                        neuron_params_e=params["neuron_params_e"], neuron_params_i=params["neuron_params_i"], 
                        stdp_synapse_params=params["stdp_synapse_params"], 
                        n_inp=params["n_inp"], n_e=params["n_e"], n_i=params["n_i"], max_rate=params["max_rate"], 
                        static_synapse_params_ei=params["static_synapse_params_ei"], static_synapse_params_ie=params["static_synapse_params_ie"],
                        network_type="WTA")
    acc, predict_labels, answer_labels, wronged_image_idx = validator.validate(n_samples=params["n_samples"])
    print(f"Accuracy: {acc}")
    print(f"Wrongly predicted images: {wronged_image_idx}")
    
    # 結果を記録
    with open(f"{SAVE_PATH}/result.txt", "w") as f:
        f.write(f"Accuracy: {acc*100}%\n")
        f.write("\n[Answer labels -> Predict labels]\n")
        for i in range(len(answer_labels)):
            f.write(f"Image {i}: {answer_labels[i]} -> {predict_labels[i]}\n")
        f.write("\n[Wrongly predicted images]\n")
        f.write("Wrong Image idx: Answer labels -> Predict labels\n")
        for idx in wronged_image_idx:
            f.write(f"Image {idx}: {answer_labels[idx]} -> {predict_labels[idx]}\n")
            
    tools.change_dir_name(SAVE_PATH, f"_validated_acc={acc*100:.2f}%")


