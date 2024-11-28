"""
既存のネットワークを使ってAccuracyを計算します。
"""
from brian2 import *
import Brian2_Framework.Datasets as mnist
from Brian2_Framework.Networks import Diehl_and_Cook_WTA, Chunk_WTA, Center_Surround_WTA, Cortex
import Brian2_Framework.Tools as tools
from brian2.units import *
import numpy as np
from tqdm import tqdm
import pickle
from Brian2_Framework.Plotters import Common_Plotter
import pprint as pp
import matplotlib.pyplot as plt
import os
import seaborn as sns
import Brian2_Framework.Plotters as Plotters
from datetime import datetime as dt
from Brian2_Framework.Plotters import Common_Plotter
import plotly.graph_objects as go
import plotly.io as pio
import shutil
import matplotlib
import time
matplotlib.use("Agg")
class Validator():
    """
    既存のネットワークを使ってAccuracyを計算するためのクラス
    """
    def __init__(self, target_path:str, assigned_labels_path:str, network_type:str, params:dict, params_mc:dict=None, labels:list=np.arange(10), enable_monitor:bool=False):
        """
        ネットワークでAccuracyを計算するValidatorを作成します。

        Args:
            target_path (str): 重みを読み込んだり，結果を保存するディレクトリのパス
            assigned_labels_path (str): 割り当てられたラベルを保存したス
            network_type (str): ネットワークの種類
            params (dict): ネットワークのパラメータ
            params_mc (dict): ミニカラムのパラメータ
            labels (list): テストデータに含まれるラベルのリスト
        Methods:
            validate(self, n_samples:int):
                ネットワークを実行してValidationを実行します。
        """
        self.target_path = target_path
        self.params = params
        self.network_type = network_type
        self.enable_monitor = enable_monitor
        self.n_labels = len(labels)
        self.labels = labels
        np.random.seed(params["seed"])
        if network_type == "WTA":
            self.model = Diehl_and_Cook_WTA(enable_monitor=enable_monitor, params=self.params) # ネットワークを作成
        elif network_type == "Chunk_WTA":
            self.model = Chunk_WTA(enable_monitor=enable_monitor, params=self.params) # ネットワークを作成
        elif network_type == "WTA_CS":
            self.model = Center_Surround_WTA(enable_monitor=enable_monitor, params=self.params) # ネットワークを作成
        elif network_type == "Cortex":
            self.model = Cortex(enable_monitor=enable_monitor, params=self.params) # ネットワークを作成
        else:
            raise ValueError("Validation用のネットワークの種類を正しくしてください。:", network_type)
        
        # 重みを読み込む
        if network_type == "WTA" or network_type == "Chunk_WTA" or network_type == "WTA_CS":
            self.weight_path = os.path.join(target_path, "weights.npy")
            with open(self.weight_path, "rb") as f:
                weights = np.load(f)
                self.model.network["S_0"].w = weights # 重みを復元
        elif network_type == "Cortex":
            for i in range(len(params_mc["n_mini_column"])):
                with open(os.path.join(target_path, f"mc{i}_weights.npy"), "rb") as f:
                    weights = np.load(f)
                    self.model.network[f"mc{i}_S_0"].w = weights
        self.model.disable_learning() # 学習を無効に
        with open(assigned_labels_path, "rb") as f: # 割り当てられたラベルを読み込む
            self.assigned_labels = pickle.load(f)

    def _predict_labels(self, interval, n_neuron:int, n_labels:int):
        """
        ニューロンの発火情報と割り当てられたラベルを見て、テスト画像のラベルを予測します。
        
        Args:
            interval (float): インターバル時間(ms)
            n_neuron (int): ニューロンの数
            n_labels (int): ラベルの数
        Returns:
            predicted_labels (list): 予測されたラベルのリスト
        """
        predicted_labels = []
        self.interval = interval / ms
        self.spikes_list = list(zip(self.model.network["spikemon_for_assign"].i, self.model.network["spikemon_for_assign"].t)) # スパイクモニターからスパイクのリストを作成
        self.spike_cnt4all = np.zeros((len(self.labels), n_neuron))
        # spike_cnt4all[image_idx][neuron_idx]
        for n, label in tqdm(enumerate(self.labels), desc="assigning labels", total=len(self.labels), dynamic_ncols=True):
            # spike_cnt = np.zeros((n_neuron)) # 一つの入力画像に対するスパイク数をカウント
            # 呈示時間を計算
            start_time = n * self.interval
            end_time = (n + 1) * self.interval
            # interval内のニューロン別のスパイク数をカウント
            
            neuron_idx = [spike[0] for spike in self.spikes_list if start_time <= spike[1]/ms < end_time] # インターバル内に発火したニューロンidxのリスト
            # neuron_idx = [18, 28, 10, ...]
            for i in neuron_idx: # インターバル内のスパイク数をニューロン別にカウント
                self.spike_cnt4all[n][i] += 1
            predicted_labels.append(int(self.assigned_labels[np.argmax(self.spike_cnt4all[n])]))
        
        return predicted_labels
    
    def validate(self, n_samples=1000, examination_name="test"):
        """
        バリデーションを実行します。

        Args:
            n_samples (int, optional): サンプル数. Defaults to 1000.
            examination_name (str, optional): 検証名. Defaults to "test".
        """
        self.images, self.labels = mnist.get_mnist_sample(n_samples, "test", self.labels)
        
        tools.print_validation_start()
        print("\n[INFO] validation name:", examination_name)
        print("[INFO] object directory:", self.target_path)
        pltr = Common_Plotter()
        if os.path.exists(os.path.join(self.target_path, "VALIDATING", "graphs")):
            while True:
                try:
                    shutil.rmtree(os.path.join(self.target_path, "VALIDATING", "graphs"))
                    break
                except Exception as e:
                    print(e)
                    print("２秒後に再試行...")
                    time.sleep(2) # 待機して再試行
        os.makedirs(os.path.join(self.target_path, "VALIDATING", "graphs"))
        cnt_fr_nonzero = np.zeros((n_samples)) # 発火率が0でないニューロンの数を入力画像ごとにカウント
        
        #! ================================= ネットワークの実行 =================================
        for i in tqdm(range(n_samples), 
                     desc="Simulating", 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
                     dynamic_ncols=False,  # 固定幅に
                     ncols=80):  # 幅を指定):
            self.model.set_input_image(self.images[i], self.params["spontaneous_rate"])
            self.model.run(self.params["exposure_time"])
            self.model.reset()

            # ヒートマップ
            if self.enable_monitor:
                os.makedirs(os.path.join(self.target_path, "VALIDATING", "graphs", "heatmap"), exist_ok=True)
                heatmap_data = pltr.firing_rate_heatmap(spikemon=self.model.network["spikemon_for_assign"], 
                                        start_time=self.params["exposure_time"]*i, 
                                        end_time=self.params["exposure_time"]*(i+1), 
                                        save_path=os.path.join(self.target_path, "VALIDATING", "graphs", "heatmap"), 
                                        n_this_fig=i)
                cnt_fr_nonzero[i] = np.sum(np.where(heatmap_data != 0, 1, 0))
                
            # 勝者ニューロンの可視化
            os.makedirs(os.path.join(self.target_path, "VALIDATING", "graphs", "wta_response"), exist_ok=True)
            pltr.visualize_wta_response(self.images[i], synapse=self.model.network["S_0"], spikemon=self.model.network["spikemon_for_assign"], start_time=self.params["exposure_time"]*i, exposure_time=self.params["exposure_time"], save_path=os.path.join(self.target_path, "VALIDATING", "graphs", "wta_response"), n_this_fig=i)
        
            
        # ============================= ラベルの予測と精度の算出 =============================
        predict_labels = self._predict_labels(interval=self.params["exposure_time"], n_neuron=self.params["n_e"], n_labels=self.n_labels)
        acc = np.sum(self.labels == predict_labels) / len(self.labels)
        
        if examination_name is None:
            validation_name = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + "validated_acc=" + f"{acc*100:.2f}%"
        else:
            validation_name = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + "_acc=" + f"{acc*100:.2f}%_" + examination_name
        
        # labels : 画像のラベル
        # predict_labels : 予測されたラベル
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name), exist_ok=True)
        print("acc:", acc)
        
        # =================== 発火率が0でないニューロンの数をプロット ===================
        if self.enable_monitor:
            plt.figure(figsize=(10, 8))
            plt.bar(range(len(self.labels)), cnt_fr_nonzero, color="black")
            plt.xticks(range(len(self.labels)), self.labels)
            # 発火数の平均値を計算
            mean_firing = np.mean(cnt_fr_nonzero)
            
            # 平均値を示す水平線を追加
            plt.axhline(y=mean_firing, color='r', linestyle='--', label=f'Mean: {mean_firing:.2f}')
            plt.legend()
            
            plt.title("Number of Neurons that Fired")
            plt.xlabel("Input Image Label")
            plt.ylabel("Number of Neurons")
            os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "fr_nonzero"), exist_ok=True)
            plt.savefig(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "fr_nonzero", "fr_nonzero.png"))
            plt.close()
        
        # =================== Winnerニューロンとnot winnerニューロンの入力電流のプロット ===================
        if self.enable_monitor:
            # 各画像ごとのプロット
            winner_Ie_all = []
            winner_Ii_all = []
            not_winner_Ie_all = []
            not_winner_Ii_all = []
            
            for i, image_idx in tqdm(enumerate(range(n_samples)), desc="plotting winner and not winner neuron currents", total=n_samples, dynamic_ncols=True):
                winner_neuron_idx = np.argmax(self.spike_cnt4all[image_idx])
                not_winner_neuron_idxs = np.argsort(self.spike_cnt4all[image_idx])[::-1]
                start_time = i * self.interval
                end_time = (i + 1) * self.interval
                
                # winner neuronの電流を取得
                winner_Ie = np.mean(self.model.network["statemon_N_1"].Ie[winner_neuron_idx, (start_time < self.model.network["statemon_N_1"].t/ms) & (self.model.network["statemon_N_1"].t/ms < end_time)])
                winner_Ii = np.mean(self.model.network["statemon_N_1"].Ii[winner_neuron_idx, (start_time < self.model.network["statemon_N_1"].t/ms) & (self.model.network["statemon_N_1"].t/ms < end_time)])
                
                winner_Ie_all.append(winner_Ie)
                winner_Ii_all.append(winner_Ii)
                
                # not winner neuronの電流を取得（上位5位を除く）
                not_winner_Ie = []
                not_winner_Ii = []
                for idx in not_winner_neuron_idxs[5:]:
                    not_winner_Ie.append(np.mean(self.model.network["statemon_N_1"].Ie[idx, (start_time < self.model.network["statemon_N_1"].t/ms) & (self.model.network["statemon_N_1"].t/ms < end_time)]))
                    not_winner_Ii.append(np.mean(self.model.network["statemon_N_1"].Ii[idx, (start_time < self.model.network["statemon_N_1"].t/ms) & (self.model.network["statemon_N_1"].t/ms < end_time)]))
                
                not_winner_Ie = np.mean(not_winner_Ie)
                not_winner_Ii = np.mean(not_winner_Ii)
                
                not_winner_Ie_all.append(not_winner_Ie)
                not_winner_Ii_all.append(not_winner_Ii)
                
                # プロット
                fig, ax1 = plt.subplots(figsize=(10, 8))
                ax2 = ax1.twinx()  # 2つ目のy軸を作成
                
                # バーの位置を設定
                x = np.array([0, 1])
                width = 0.35
                
                # 興奮性電流（Ie）のプロット
                rects1 = ax1.bar(x - width/2, [winner_Ie, not_winner_Ie], width, 
                                color="red", alpha=0.6, label="Excitatory (Ie)")
                
                # 抑制性電流（Ii）のプロット（正の値のまま表示）
                rects2 = ax2.bar(x + width/2, [winner_Ii, not_winner_Ii], width,
                                color="blue", alpha=0.6, label="Inhibitory (Ii)")
                
                # 軸の設定
                ax1.set_ylabel("Excitatory Current (pA)", color="red")
                ax2.set_ylabel("Inhibitory Current (pA)", color="blue")
                ax1.tick_params(axis='y', labelcolor="red")
                ax2.tick_params(axis='y', labelcolor="blue")
                
                # y軸の最大値の設定
                _, y_max = ax1.get_ylim()
                y_min, _ = ax2.get_ylim()
                y_max = max(y_max, -y_min)
                
                ax1.set_ylim(0, y_max)
                ax2.set_ylim(0, -y_max)  # 上限と下限を逆にして設定
                
                # x軸のラベル
                plt.xticks(x, ["Winner", "Not Winner"])
                
                # タイトルと凡例
                ax1.set_title(f"Current Comparison (Image {image_idx}, Label {self.labels[image_idx]})")
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                # グリッドの追加（主軸のみ）
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                # 保存
                os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "currents"), exist_ok=True)
                plt.savefig(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "currents", f"current_comparison_{image_idx}.png"))
                plt.close()
            
            # 全画像の平均をプロット
            fig, ax1 = plt.subplots(figsize=(10, 8))
            ax2 = ax1.twinx()
            
            x = np.array([0, 1])
            width = 0.35
            
            winner_Ie_mean = np.mean(winner_Ie_all)
            winner_Ii_mean = np.mean(winner_Ii_all)
            not_winner_Ie_mean = np.mean(not_winner_Ie_all)
            not_winner_Ii_mean = np.mean(not_winner_Ii_all)
            
            # 標準偏差も計算
            winner_Ie_std = np.std(winner_Ie_all)
            winner_Ii_std = np.std(winner_Ii_all)
            not_winner_Ie_std = np.std(not_winner_Ie_all)
            not_winner_Ii_std = np.std(not_winner_Ii_all)
            
            # 平均値のプロット（エラーバー付き）
            ax1.bar(x - width/2, [winner_Ie_mean, not_winner_Ie_mean], width, 
                   yerr=[winner_Ie_std, not_winner_Ie_std],
                   color="red", alpha=0.6, label="Excitatory (Ie)")
            ax2.bar(x + width/2, [winner_Ii_mean, not_winner_Ii_mean], width,
                   yerr=[winner_Ii_std, not_winner_Ii_std],
                   color="blue", alpha=0.6, label="Inhibitory (Ii)")
            
            ax1.set_ylabel("Excitatory Current (pA)", color="red")
            ax2.set_ylabel("Inhibitory Current (pA)", color="blue")
            ax1.tick_params(axis='y', labelcolor="red")
            ax2.tick_params(axis='y', labelcolor="blue")
            
            _, y_max = ax1.get_ylim()
            y_min, _ = ax2.get_ylim()
            y_max = max(y_max, -y_min)
            
            ax1.set_ylim(0, y_max)
            ax2.set_ylim(0, -y_max)
            
            plt.xticks(x, ["Winner", "Not Winner"])
            
            ax1.set_title("Average Current Comparison (All Images)")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "currents", "current_comparison_average.png"))
            plt.close()


        # validationパラメータを保存
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name), exist_ok=True)
        tools.save_parameters(os.path.join(self.target_path, "VALIDATING", validation_name, "validation_params.json"), self.params)
            
        # モニターを保存
        if self.enable_monitor:
            os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "monitors"), exist_ok=True)
            if self.network_type == "WTA":
                tools.save_monitor(self.model.network["spikemon_inp"], os.path.join(self.target_path, "VALIDATING", validation_name, "monitors", "spikemon_inp.pkl"))
                tools.save_monitor(self.model.network["spikemon_N_1"], os.path.join(self.target_path, "VALIDATING", validation_name, "monitors", "spikemon_N_1.pkl"))
                tools.save_monitor(self.model.network["spikemon_N_2"], os.path.join(self.target_path, "VALIDATING", validation_name, "monitors", "spikemon_N_2.pkl"))
                tools.save_monitor(self.model.network["statemon_N_1"], os.path.join(self.target_path, "VALIDATING", validation_name, "monitors", "statemon_N_1.pkl"), ["v", "Ie", "Ii", "ge", "gi"])
                tools.save_monitor(self.model.network["statemon_N_2"], os.path.join(self.target_path, "VALIDATING", validation_name, "monitors", "statemon_N_2.pkl"), ["v", "Ie", "Ii", "ge", "gi"])

        
        # =================== 予測ラベルと正解ラベルをテキストで保存 ===================
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name), exist_ok=True)
        wronged_image_idx = list(np.where(self.labels != predict_labels)[0]) # 不正解画像のインデックスを取得
        with open(os.path.join(self.target_path, "VALIDATING", validation_name, "result.txt"), "w") as f:
            f.write(f"Accuracy: {acc*100}%\n")
            f.write("\n[Answer labels -> Predict labels]\n")
            for i in range(len(self.labels)):
                f.write(f"Image {i}: {self.labels[i]} -> {predict_labels[i]}\n")
            f.write("\n[Wrongly predicted images]\n")
            f.write("Wrong Image idx: Answer labels -> Predict labels\n")
            for idx in wronged_image_idx:
                f.write(f"Image {idx}: {self.labels[idx]} -> {predict_labels[idx]}\n")
        
        # ========================== 不正解画像を保存 ==========================
        for i in range(10):
            os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "wrong_images", f"class_{i}"), exist_ok=True)
        for idx in wronged_image_idx:
            plt.imsave(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "wrong_images", f"class_{self.labels[idx]}", f"wrong_image_{idx}.png"), self.images[idx], cmap="gray")
            
        # ================ 予測ラベルと正解ラベルのConfusion Matrixを保存 ================
        confusion_matrix = np.zeros((10, 10))
        for i in range(len(self.labels)):
            confusion_matrix[self.labels[i]][predict_labels[i]] += 1 # confusion_matrix[正解ラベル][予測ラベル]
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix.astype(int), annot=True, fmt='d', cmap='plasma')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "confusion_matrix.png"))
        plt.close()
        
        # ============= 不正解の画像にたいしてどのRFを持つニューロンが発火したかランキングを保存 =============
            
        # 不正解の画像に対してどのような重みを持つニューロンが発火したかを可視化
        for i in range(10):
            os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "top 10 wrong weights against each image", f"class_{i}"), exist_ok=True)
        for idx in tqdm(wronged_image_idx, desc="saving top 10 wrong weights against each image", dynamic_ncols=True):
            spike_counts = self.spike_cnt4all[idx]
            # spike_cnt4all[image_idx][neuron_idx]

            # 発火数の多い上位10個のニューロンのインデックスを取得
            top_neurons = np.argsort(spike_counts)[::-1]
            
            # 発火数が同じニューロンを昇順に並び替え
            sorted_neurons = []
            for count in sorted(set(spike_counts[top_neurons]), reverse=True):
                same_count_neurons = [n for n in top_neurons if spike_counts[n] == count]
                sorted_neurons.extend(sorted(same_count_neurons))
            top_10_neurons = sorted_neurons[:10]
            
            # プロットの準備
            fig, axes = plt.subplots(3, 5, figsize=(20, 8))
            fig.suptitle(f"Top 10 Neuron Weights for Incorrect Prediction (Image {idx})\n"
                         f"True: {self.labels[idx]}, Predicted: {predict_labels[idx]}", fontsize=16)
            im = axes[0, 0].imshow(self.images[idx], cmap='gray')
            axes[0, 0].axis('off')
            axes[0, 0].set_title(f"Input Image\nlabel: {self.labels[idx]}")
            for i in range(5):
                axes[0, i].axis('off')
            for i, neuron_idx in enumerate(top_10_neurons):
                row = i // 5 + 1
                col = i % 5
                
                
                axes[row, col].axis('off')
                weight = self.model.network["S_0"].w[:, neuron_idx].reshape(28, 28)
                # if spike_counts[neuron_idx] == 0:
                #     continue
                im = axes[row, col].imshow(weight, cmap='viridis')
                axes[row, col].set_title(f"Neuron {neuron_idx}\nSpikes: {spike_counts[neuron_idx]}\nassigned label: {self.assigned_labels[neuron_idx]}")
                fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "top 10 wrong weights against each image", f"class_{self.labels[idx]}", f"true_{self.labels[idx]}_predicted_{predict_labels[idx]}_image_{idx}.png"))
            plt.close()
            
        # ================== 任意のRFがどの画像に対して発火したかランキングを保存 ==================
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "top 10 images neurons fire"), exist_ok=True)
        for idx in tqdm(range(self.params["n_e"]), desc="saving top 10 images neurons fire", dynamic_ncols=True):
            spike_counts = self.spike_cnt4all[:, idx]
            # spike_cnt4all[image_idx][neuron_idx]
            top_10_images = np.argsort(spike_counts)[-10:][::-1]
            
            fig, axes = plt.subplots(3, 5, figsize=(20, 8))
            fig.suptitle(f"Top 10 Images for Neuron {idx}\n"
                         f"neuron label: {self.assigned_labels[idx]}", fontsize=16)
            weight = self.model.network["S_0"].w[:, idx].reshape(28, 28)
            im = axes[0, 0].imshow(weight, cmap='viridis')
            axes[0, 0].axis('off')
            for i in range(5):
                axes[0, i].axis('off')
            for i, image_idx in enumerate(top_10_images):
                row = i // 5 + 1
                col = i % 5
                axes[row, col].axis('off')
                axes[row, col].imshow(self.images[image_idx], cmap='gray')
            plt.tight_layout()
            plt.savefig(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "top 10 images neurons fire", f"top_10_images_neuron_{idx}.png"))
            plt.close()
            
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs", "heatmap"), exist_ok=True)
        tools.copy_directory(os.path.join(self.target_path, "VALIDATING", "graphs"), os.path.join(self.target_path, "VALIDATING", validation_name, "graphs"))
            
        if self.enable_monitor:
            return os.path.join(self.target_path, "VALIDATING", validation_name), mean_firing
        else:
            return os.path.join(self.target_path, "VALIDATING", validation_name)
