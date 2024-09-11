from brian2 import *
import os
import glob
from PIL import Image
import json
import pprint as pp
import random
import pickle as pkl
from tqdm import tqdm

def normalize_weight(synapse, goal_sum_weight, n_i, n_j):
    """
    重みの合計値をgoal_sum_weightに正規化する

    Args:
        synapses (Synapses): Synapsesオブジェクトのリスト

    Returns:
        None
    """
    connections = np.zeros((n_i, n_j))
    connections[synapse.i, synapse.j] = synapse.w
    col_sum = np.sum(connections, axis=0)
    col_factor = goal_sum_weight / col_sum
    for i in range(n_j):
        connections[:, i] *= col_factor[i]
    synapse.w = connections[synapse.i, synapse.j]
        
def make_gif(fps=20, inp_dir="", out_dir="", out_name="output.gif"):
    """
    画像をGIFに変換する
    """

    # 指定されたディレクトリから画像ファイルを取得
    image_files = sorted(glob.glob(os.path.join(inp_dir, "*.png")), key=lambda x: os.path.getmtime(x))
    if not image_files:
        print("指定されたディレクトリに画像ファイルが見つかりません。")
        return
    
    # 画像を開く
    images = []
    for file in image_files:
        img = Image.open(file)
        images.append(img)
    
    # 重複する最初の画像を削除
    unique_images = []
    prev_image = None
    for img in images:
        if img != prev_image:
            unique_images.append(img)
            prev_image = img
    
    # GIFファイルを保存
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, out_name)
    unique_images[0].save(output_path, save_all=True, append_images=unique_images[1:], duration=1000//fps, loop=0)
    
    print(f"\tGIF image saved: {output_path}")
    
def normalize_image_by_sum(image):
    """
    画像を画素値の合計で正規化する関数
    """
    total_sum = image.sum()
    if total_sum != 0:
        return image / total_sum 
    else:
        return image  # 合計がゼロの場合、正規化を行わない

    
def adjust_firing_rate(image, target_mean_rate, max_rate):
    """
    入力ニューロンの平均発火率を統一するために画像をスケーリングする関数
    """
    total_sum = image.sum()
    if total_sum != 0:
        scaling_factor = (target_mean_rate * image.size) / (total_sum * max_rate)
        adjusted_image = image * scaling_factor
        return adjusted_image
    else:
        return image
    
def convert_quantity(obj):
    """
    Quantityオブジェクトを数値に変換する関数
    """
    if isinstance(obj, Quantity):
        return float(obj)  # 単位を無視して数値のみを返す
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # ndarrayをリストに変換
    else:
        return str(obj)  # その他のオブジェクトは文字列に変換
    
def print_firing_rate(spikemon):
    """
    スパイクモニターからニューロンごとの発火率を取得する
    """
    # スパイクモニターからニューロンごとの発火率を計算
    spike_counts = spikemon.count
    duration = spikemon.t[-1] - spikemon.t[0]
    firing_rates = spike_counts / (duration / second)
    
    # 発火率を表示
    print("ニューロンごとの発火率 (Hz):")
    for i, rate in enumerate(firing_rates):
        print(f"ニューロン {i}: {rate:.2f} Hz")
    
    # 平均発火率を計算して表示
    average_rate = np.mean(firing_rates)
    print(f"\n平均発火率: {average_rate:.2f} Hz")
    
    return firing_rates

def assign_labels2neurons(spikemon, n_neuron:int, n_labels:int, input_labels:list, presentation_time, reset_time):
    """
    ニューロンにラベルを割り当てます。

    Args:
        spikemon (SpikeMonitor): スパイクモニター
        n_neuron (int): ニューロンの数
        n_labels (int): ラベルの種類数
        input_labels (list): 学習で使用された画像のラベルのリスト（入力された順番）
        presentation_time (float): 画像提示時間(ms)
        reset_time (float): 画像リセット時間(ms)
    Returns:
        assigned_labels (list): ニューロンに割り当てられたラベルのリスト
    """
    presentation_time /= ms
    reset_time /= ms
    interval_time = presentation_time + reset_time
    spikes = list(zip(spikemon.i, spikemon.t))
    spike_cnt = np.zeros((n_neuron, n_labels))
    for n, label in tqdm(enumerate(input_labels), desc="assigning labels", total=len(input_labels)):
        start_time = n * interval_time
        end_time = (n + 1) * interval_time
        neuron_idx = [spike[0] for spike in spikes if start_time <= spike[1]/ms < end_time]
        spike_cnt[neuron_idx, label] += 1

    assigned_labels = np.full(n_neuron, -1)  # デフォルトは-1で初期化
    for i in range(n_neuron):
        max_indices = np.where(spike_cnt[i] == spike_cnt[i].max())[0]
        if max_indices.size > 0:
            assigned_labels[i] = np.random.choice(max_indices)
        else:
            print(f"警告: ニューロン {i} のスパイク数が全て0です")
            print(f"NaN の有無: {np.isnan(spike_cnt[i]).any()}")
            print(f"inf の有無: {np.isinf(spike_cnt[i]).any()}")
    
    return assigned_labels


def reset_network(network):
    """
    ネットワークをリセットする
    """
    # ニューロン
    neurons = [obj for obj in network.objects if isinstance(obj, NeuronGroup)]

    # シナプス
    synapses = [obj for obj in network.objects if isinstance(obj, Synapses)]
    
    for i in range(len(neurons)):
        neurons[i].v = -65
        
    for i in range(len(synapses)):
        synapses[i].ge = 0
        synapses[i].gi = 0
        try:
            synapses[i].apre = 0
            synapses[i].apost = 0
        except:
            pass



        




