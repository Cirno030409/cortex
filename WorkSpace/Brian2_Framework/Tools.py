from brian2 import *
import os
import glob
from PIL import Image
import json
import pprint as pp
import random
import pickle as pkl
from tqdm import tqdm
import time
import shutil

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
    
    Args:
        fps (int): フレームレート
        inp_dir (str): 入力ディレクトリ
        out_dir (str): 出力ディレクトリ
        out_name (str): 出力ファイル名
    """

    all_files = os.listdir(inp_dir)
    pattern = re.compile(r'^[0-9]+\.png$')
    image_files = [file for file in all_files if pattern.match(file)]
    image_files = [os.path.join(inp_dir, file) for file in image_files]
    image_files = sorted(image_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
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
    Quantityオブジェクトを文字列に変換する関数
    """
    if isinstance(obj, Quantity):
        value = str(obj)
        units = ["ms", "second", "ks", "Hz", "kHz", "mV", "V", "nS", "uS", "S", "nF", "uF", "F"]
        for unit in units:
            value = value.replace(unit, f"*{unit}")
        return value  # Quantityオブジェクトを文字列に変換
    elif isinstance(obj, np.ndarray):
        raise ValueError("ndarrayは変換できません")
    else:
        return str(obj)  # その他のオブジェクトは文字列に変換
    
def get_spikes_within_time_range(spikemon, start_time, end_time):
    """
    スパイクモニターから指定された時間範囲内のスパイクを取得する
    
    Args:
        spikemon (SpikeMonitor): スパイクモニター
        start_time (float): 開始時間(ms)
        end_time (float): 終了時間(ms)
    Returns:
        spikes (list): スパイクのリスト
            spikes = [spikemon.t, spikemon.i]
    """
    spikes = list(zip(spikemon.t, spikemon.i))
    spikes = [spike for spike in spikes if start_time <= spike[0] < end_time] # 単位そのままで比較
    return spikes


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

def change_dir_name(dir_path, add_name):
    """
    ディレクトリ名を変更する
    
    Args:
        dir_path (str): 変更するディレクトリのパス
        new_name (str): 新しいディレクトリ名
    Returns:
        None
    """
    time.sleep(1)
    # 完了したのでディレクトリ名を変更
    new_save_path = dir_path[:-1] + add_name
    # print("dir_path: ", dir_path)
    # print("new_save_path: ", new_save_path)
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)
    for filename in os.listdir(dir_path):
        old_file = os.path.join(dir_path, filename)
        new_file = os.path.join(new_save_path, filename)
        os.rename(old_file, new_file)
    time.sleep(1)
    shutil.rmtree(dir_path)
    print(f"[INFO] ディレクトリ名を {new_save_path} に変更しました。")
    return new_save_path

def get_firing_rate(spikemon, start_time=None, end_time=None, enable_print:bool=False):
    """
    スパイクモニターからニューロンごとの発火率を取得する
    
    Args:
        spikemon (SpikeMonitor): スパイクモニター
        start_time (float): 開始時間(ms)
        end_time (float): 終了時間(ms)
        enable_print (bool): 発火率を表示するか
    Returns:
        firing_rates (np.ndarray): ニューロンごとの発火率
    """
    # spikemon.count[n_neurons]
    # スパイクモニターからニューロンごとの発火率を計算
    if start_time is None and end_time is None:
        spike_counts = spikemon.count
        duration = spikemon.t.max()
    else:
        duration = end_time - start_time
        spikes = get_spikes_within_time_range(spikemon, start_time, end_time)
        spikes = [spike[1] for spike in spikes]
        spike_counts = np.bincount(spikes, minlength=len(spikemon.count))
    
    firing_rates = spike_counts / (duration / second)
    
    # 発火率を表示
    if enable_print:
        print("ニューロンごとの発火率 (Hz):")
        for i, rate in enumerate(firing_rates):
            print(f"ニューロン {i}: {rate:.2f} Hz")

    average_rate = np.mean(firing_rates)
    
    # 平均発火率を計算して表示
    if enable_print:
        print(f"\n平均発火率: {average_rate:.2f} Hz")
    
    return firing_rates


def reset_network(network):
    """
    ネットワークをリセットします。
    """
    # ニューロン
    neurons = [obj for obj in network.objects if isinstance(obj, NeuronGroup)]

    # シナプス
    synapses = [obj for obj in network.objects if isinstance(obj, Synapses)]
    
    for i in range(len(neurons)):
        neurons[i].v = -60
        
    for i in range(len(synapses)):
        synapses[i].ge = 0
        synapses[i].gi = 0
        try:
            synapses[i].apre = 0
            synapses[i].apost = 0
        except:
            pass

def load_parameters(file_path:str):
    """
    JSONファイルからパラメータを読み込みます。Brian2の単位変換も行います。

    Args:
        file_path (str): JSONファイルのパス
    Returns:
        parameters (dict): パラメータの辞書
    """
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    for key_name in loaded_data.keys():
        if isinstance(loaded_data[key_name], dict):
            for key, value in loaded_data[key_name].items():
                if isinstance(value, str) and '*' in value:
                    value, unit = value.split('*')
                    loaded_data[key_name][key] = float(value) * eval(unit)
        else:
            if isinstance(loaded_data[key_name], str) and '*' in loaded_data[key_name]:
                value, unit = loaded_data[key_name].split('*')
                loaded_data[key_name] = float(value) * eval(unit)
    return loaded_data

def save_parameters(dir_path:str, parameters:dict):
    """
    JSONファイルにパラメータを保存します。Brian2の単位変換も行います。

    Args:
        dir_path (str): ディレクトリのパス
        parameters (dict): パラメータの辞書
    Returns:
        None
    """
    with open(dir_path + "parameters.json", "w") as f:
        json.dump(parameters, f, indent=4, default=convert_quantity)
        
def memo_assigned_labels(save_path, assigned_labels):
    """
    ニューロンに割り当てられたラベルを読みやす形でテキストファイルに保存します。
    """
    with open(save_path + "assigned_labels.txt", "w") as f:
        f.write("[Assigned labels]\n")
        for i in range(len(assigned_labels)):
            f.write(f"\tneuron {i}: {assigned_labels[i]}\n")
            
def save_assigned_labels(save_path, assigned_labels):
    """
    ニューロンに割り当てられたラベルをValidate時に使用するためにpklファイルに保存します。
    """
    with open(save_path + "assigned_labels.pkl", "wb") as f:
        pkl.dump(assigned_labels, f)



