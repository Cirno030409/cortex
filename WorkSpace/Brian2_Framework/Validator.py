"""
既存のネットワークを使ってAccuracyを計算します。
"""
from brian2 import *
import Brian2_Framework.Mnist as mnist
from Brian2_Framework.Networks import Diehl_and_Cook_WTA, Chunk_WTA
import Brian2_Framework.Tools as tools
from brian2.units import *
import numpy as np
from tqdm import tqdm
import pickle
from Brian2_Framework.Plotters import Common_Plotter
import pprint as pp
class Validator():
    
    def __init__(self, weight_path:str, assigned_labels_path:str, network_type:str, **kwargs):
        """
        ネットワークでAccuracyを計算するValidatorを作成します。

        Args:
            weight_path (str): _description_
            answer_assigned_labels (list): _description_
        """
        if network_type == "WTA":
            self.model = Diehl_and_Cook_WTA(enable_monitor=False, **kwargs) # ネットワークを作成
        elif network_type == "Chunk_WTA":
            self.model = Chunk_WTA(enable_monitor=False, **kwargs) # ネットワークを作成
        else:
            raise ValueError("Invalid network type")
        with open(weight_path, "rb") as f: # 重みを読み込む
            weights = np.load(f)
        self.model.network["S_0"].w = weights # 重みを復元
        self.model.disable_learning() # 学習を無効に
        with open(assigned_labels_path, "rb") as f:
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
        spikes_list = list(zip(self.model.network["spikemon_for_assign"].i, self.model.network["spikemon_for_assign"].t)) # スパイクモニターからスパイクのリストを作成
        for n, label in tqdm(enumerate(self.labels), desc="assigning labels", total=len(self.labels), dynamic_ncols=True):
            spike_cnt = np.zeros((n_neuron)) # 一つの入力画像に対するスパイク数をカウント
            # 呈示時間を計算
            start_time = n * interval
            end_time = (n + 1) * interval
            # interval内のニューロン別のスパイク数をカウント
            neuron_idx = [spike[0] for spike in spikes_list if start_time <= spike[1]/ms < end_time] # インターバル内に発火したニューロンidxのリスト
            # neuron_idx = [18, 28, 10, ...]
            for i in neuron_idx: # インターバル内のスパイク数をニューロン別にカウント
                spike_cnt[i] += 1
            predicted_labels.append(int(self.assigned_labels[np.argmax(spike_cnt)]))
        
        return predicted_labels
    
    def _get_accuracy(self, answer_labels, learned_labels):
        """
        割り当てられたラベルと入力されたラベルの精度を計算する
        """
        return np.sum(answer_labels == learned_labels) / len(answer_labels), list(np.where(answer_labels != learned_labels)[0])
    
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
        image, self.labels = mnist.get_mnist_sample(n_samples=n_samples, dataset='test') # テストデータを取得
        print("[PROCESS] Validation started.")
        for i in tqdm(range(n_samples), desc="simulating", dynamic_ncols=True):
            self.model.change_image(image[i])
            self.model.network.run(150*ms)
            tools.reset_network(self.model.network)
        
        predict_labels = self._predict_labels(interval=150*ms, n_neuron=100, n_labels=10)
        acc, wronged_image_idx = self._get_accuracy(self.labels, predict_labels)
        
        print(f"[INFO] Accuracy: {acc}")
        if n_samples <= 100:
            print(f"[INFO] Predicted labels: ")
            for i in range(len(predict_labels)):
                print(f"\tImage {i}: {predict_labels[i]}")
            print(f"[INFO] True labels: ")
            for i in range(len(self.labels)):
                print(f"\tImage {i}: {self.labels[i]}")
                
        return acc, predict_labels, self.labels, wronged_image_idx
        











        







