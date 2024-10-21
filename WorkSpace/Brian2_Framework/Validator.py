"""
既存のネットワークを使ってAccuracyを計算します。
"""
from brian2 import *
import Brian2_Framework.Datasets as mnist
from Brian2_Framework.Networks import Diehl_and_Cook_WTA, Chunk_WTA, Center_Surround_WTA
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
class Validator():
    """
    既存のネットワークを使ってAccuracyを計算するためのクラス
    """
    def __init__(self, target_path:str, assigned_labels_path:str, network_type:str, params_json_path:str):
        """
        ネットワークでAccuracyを計算するValidatorを作成します。

        Args:
            target_path (str): 重みを読み込んだり，結果を保存するディレクトリのパス
            assigned_labels_path (str): 割り当てられたラベルを保存したパス
            network_type (str): ネットワークの種類
            params_json_path (str): ネットワークのパラメータを保存したjsonファイルのパス
            
        Methods:
            validate(self, n_samples:int):
                ネットワークを実行してValidationを実行します。
        """
        self.target_path = target_path
        self.weight_path = os.path.join(target_path, "weights.npy")
        self.params = tools.load_parameters(params_json_path)
        if network_type == "WTA":
            self.model = Diehl_and_Cook_WTA(enable_monitor=False, params_json_path=params_json_path) # ネットワークを作成
        elif network_type == "Chunk_WTA":
            self.model = Chunk_WTA(enable_monitor=False, params_json_path=params_json_path) # ネットワークを作成
        elif network_type == "WTA_CS":
            self.model = Center_Surround_WTA(enable_monitor=False, params_json_path=params_json_path) # ネットワークを作成
        else:
            raise ValueError("Validation用のネットワークの種類を正しくしてください。:", network_type)
        
        with open(self.weight_path, "rb") as f: # 重みを読み込む
            weights = np.load(f)
        self.model.network["S_0"].w = weights # 重みを復元
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
        interval = interval / ms
        self.spikes_list = list(zip(self.model.network["spikemon_for_assign"].i, self.model.network["spikemon_for_assign"].t)) # スパイクモニターからスパイクのリストを作成
        self.spike_cnt4all = np.zeros((len(self.labels), n_neuron))
        # spike_cnt4all[image_idx][neuron_idx]
        for n, label in tqdm(enumerate(self.labels), desc="assigning labels", total=len(self.labels), dynamic_ncols=True):
            # spike_cnt = np.zeros((n_neuron)) # 一つの入力画像に対するスパイク数をカウント
            # 呈示時間を計算
            start_time = n * interval
            end_time = (n + 1) * interval
            # interval内のニューロン別のスパイク数をカウント
            neuron_idx = [spike[0] for spike in self.spikes_list if start_time <= spike[1]/ms < end_time] # インターバル内に発火したニューロンidxのリスト
            # neuron_idx = [18, 28, 10, ...]
            for i in neuron_idx: # インターバル内のスパイク数をニューロン別にカウント
                self.spike_cnt4all[n][i] += 1
            predicted_labels.append(int(self.assigned_labels[np.argmax(self.spike_cnt4all[n])]))
        
        return predicted_labels
    
    def validate(self, n_samples:int):
        """
        検証用のネットワークを実行してAccuracyを計算します。
        
        Args:
            n_samples (int): テストデータの数
        Returns:
            acc (float): 精度
            predict_labels (list): 予測されたラベルのリスト
            answer_labels (list): 正解のラベルのリスト
            wronged_image_idx (list): 予測が間違えた画像のインデックスのリスト
        """
        self.images, self.labels = mnist.get_mnist_sample_equality_labels(n_samples, "test")
        
        # ===================================== ネットワークの実行 ==========================================
        print("[PROCESS] Validating...")
        for i in tqdm(range(n_samples), desc="simulating", dynamic_ncols=True):
            self.model.change_image(self.images[i])
            self.model.network.run(self.params["exposure_time"])
            tools.reset_network(self.model.network)
            
        # ===================================== ラベルの予測と精度の算出 ===================================
        predict_labels = self._predict_labels(interval=self.params["exposure_time"], n_neuron=self.params["n_e"], n_labels=10)
        acc = np.sum(self.labels == predict_labels) / len(self.labels)
        print("acc:", acc)


        # ===================================== ディレクトリの作成 ===================================
        validation_name = dt.now().strftime("%Y_%m_%d_%H_%M_%S_") + "validated_acc=" + f"{acc*100:.2f}%"
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "wrong_images"), exist_ok=True)
        for i in range(10):
            os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, f"wrong_images/class_{i}"), exist_ok=True)
        for i in range(10):
            os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, f"top 10 wrong weights against each image/class_{i}"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "top 10 images neurons fire"), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, "VALIDATING", validation_name, "graphs"), exist_ok=True)
        
        tools.save_parameters(self.target_path + f"/VALIDATING/{validation_name}/", self.params)
        
        # ===================================== 予測ラベルと正解ラベルをテキストで保存 ===================================
        wronged_image_idx = list(np.where(self.labels != predict_labels)[0]) # 不正解画像のインデックスを取得
        with open(f"{self.target_path}/VALIDATING/{validation_name}/result.txt", "w") as f:
            f.write(f"Accuracy: {acc*100}%\n")
            f.write("\n[Answer labels -> Predict labels]\n")
            for i in range(len(self.labels)):
                f.write(f"Image {i}: {self.labels[i]} -> {predict_labels[i]}\n")
            f.write("\n[Wrongly predicted images]\n")
            f.write("Wrong Image idx: Answer labels -> Predict labels\n")
            for idx in wronged_image_idx:
                f.write(f"Image {idx}: {self.labels[idx]} -> {predict_labels[idx]}\n")
        
        # ===================================== 不正解画像を保存 ===================================
        for idx in wronged_image_idx:
            plt.imsave(self.target_path + f"/VALIDATING/{validation_name}/wrong_images/class_{self.labels[idx]}/wrong_image_{idx}.png", self.images[idx], cmap="gray")
            
        # ===================================== 予測ラベルと正解ラベルのConfusion Matrixを保存 ===================================
        confusion_matrix = np.zeros((10, 10))
        for i in range(len(self.labels)):
            confusion_matrix[self.labels[i]][predict_labels[i]] += 1 # confusion_matrix[正解ラベル][予測ラベル]
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix.astype(int), annot=True, fmt='d', cmap='plasma')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(self.target_path + f"/VALIDATING/{validation_name}/graphs/confusion_matrix.png")
        plt.close()
        
        # =============================== 不正解の画像にたいしてどのRFを持つニューロンが発火したかランキングを保存 ===========================
            
        # 不正解の画像に対してどのような重みを持つニューロンが発火したかを可視化
        for idx in tqdm(wronged_image_idx, desc="saving top 10 wrong weights against each image", dynamic_ncols=True):
            spike_counts = self.spike_cnt4all[idx]
            # spike_cnt4all[image_idx][neuron_idx]

            # 発火数の多い上位10個のニューロンのインデックスを取得
            top_10_neurons = np.argsort(spike_counts)[-10:][::-1]
            
            # プロットの準備
            fig, axes = plt.subplots(3, 5, figsize=(20, 8))
            fig.suptitle(f"Top 10 Neuron Weights for Incorrect Prediction (Image {idx})\n"
                         f"True: {self.labels[idx]}, Predicted: {predict_labels[idx]}", fontsize=16)
            im = axes[0, 0].imshow(self.images[idx], cmap='gray')
            axes[0, 0].axis('off')
            for i in range(5):
                axes[0, i].axis('off')
            for i, neuron_idx in enumerate(top_10_neurons):
                row = i // 5 + 1
                col = i % 5
                
                
                axes[row, col].axis('off')
                weight = self.model.network["S_0"].w[:, neuron_idx].reshape(28, 28)
                if spike_counts[neuron_idx] == 0:
                    continue
                im = axes[row, col].imshow(weight, cmap='viridis')
                axes[row, col].set_title(f"Neuron {neuron_idx}\nSpikes: {spike_counts[neuron_idx]}")
                fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(f"{self.target_path}/VALIDATING/{validation_name}/top 10 wrong weights against each image/class_{self.labels[idx]}/top_10_weights_image_{idx}.png")
            plt.close()
            
        # ================================ 任意のRFがどの画像に対して発火したかランキングを保存 ================================
        for idx in tqdm(range(self.params["n_e"]), desc="saving top 10 images neurons fire", dynamic_ncols=True):
            spike_counts = self.spike_cnt4all[:, idx]
            # spike_cnt4all[image_idx][neuron_idx]
            top_10_images = np.argsort(spike_counts)[-10:][::-1]
            
            fig, axes = plt.subplots(3, 5, figsize=(20, 8))
            fig.suptitle(f"Top 10 Images for Neuron {idx}\n"
                         f"neuron label: {self.labels[idx]}", fontsize=16)
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
            plt.savefig(f"{self.target_path}/VALIDATING/{validation_name}/top 10 images neurons fire/top_10_images_neuron_{idx}.png")
            plt.close()
            
            
            
                
        return acc
        











        







