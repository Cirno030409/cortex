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
import numpy as np
import re
from Brian2_Framework.Monitors import SpikeMonitorData, StateMonitorData
from brian2 import SpikeMonitor

def print_simulation_start():
    """
    シミュレーション開始時のメッセージを表示する
    """
    print("\n\n\n########################################################################################")
    print("#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"#                             Simulation has started!                                  #")
    print(f"#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"########################################################################################")
    
def print_validation_start():
    """
    検証開始時のメッセージを表示する
    """
    print("\n\n\n########################################################################################")
    print("#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"#                             Validation has started!                                  #")
    print(f"#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"#                                                                                      #")
    print(f"########################################################################################")

def normalize_weight(synapse, value=None, value2=None, method:str="sum") -> None:
    """
    重みを正規化します。methodには"sum", "sum_square", "max"を指定します。
    
    method:sum -> 重みの合計値がvalueになるように正規化
    method:sum_square -> 重みの二乗の合計値がvalueになるように正規化
    method:minmax -> 重みの最小値と最大値をそれぞれvalue, value2になるように正規化
    method:minus -> すべての重みからvalueを減算し，最小値が-valueになるように正規化
    Args:
        synapses (Synapses): Synapsesオブジェクトのリスト

    Returns:
        None
    """
    n_i = synapse.N_incoming_post[0]
    n_j = synapse.N_outgoing_pre[0]
    connections = np.zeros((n_i, n_j))
    connections[synapse.i, synapse.j] = synapse.w
    
    if method == "sum":
        col_sum = np.sum(connections, axis=0)
        col_factor = value / col_sum
        for i in range(n_j):
            connections[:, i] *= col_factor[i]
    elif method == "sum_square":
        col_sum = np.sum(connections, axis=0)**2
        col_factor = value / col_sum #!
        for i in range(n_j):
            connections[:, i] *= col_factor[i]
    elif method == "minmax":
        col_max = np.max(connections, axis=0)
        col_min = np.min(connections, axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1
    elif method == "minus":
        connections -= value

    else:
        raise ValueError("methodには'sum', 'sum_square', 'max'のいずれかを指定してください。")
    synapse.w = connections[synapse.i, synapse.j]
        
def make_gif(fps=20, inp_dir="", out_dir="", out_name="output.gif"):
    """
    画像をGIFに変換する
    
    Args:
        fps (int): フレームレート
        inp_dir (str): 入力ディレクトリ
        out_dir (str): 出力ィレクトリ
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
    スパイクモニターから指定され時間範囲内のスパイク取得する
    
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
    """
    presentation_time /= ms
    reset_time /= ms
    interval_time = presentation_time + reset_time
    
    # スパイクデータをNumPy配列に変換
    spike_times = spikemon.t/ms if hasattr(spikemon.t, 'dim') else spikemon.t
    spike_indices = spikemon.i
    
    # 結果を格納する配列を初期化
    spike_cnt = np.zeros((n_neuron, n_labels))
    
    # 各画像の時間間隔を計算
    for n, label in enumerate(input_labels):
        start_time = n * interval_time
        end_time = (n + 1) * interval_time
        
        # 時間窓内のスパイクを一括で特定
        time_mask = (spike_times >= start_time) & (spike_times < end_time)
        active_neurons = spike_indices[time_mask]
        
        # 発火カウントを更新
        np.add.at(spike_cnt, (active_neurons, label), 1)
    
    return np.argmax(spike_cnt, axis=1)

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
    time.sleep(3)
    shutil.rmtree(dir_path)
    print(f"[INFO] ディレクトリ名を {new_save_path} に変更しました。")
    return new_save_path

def copy_directory(src_dir, dst_dir):
    """
    ディレクトリを再帰的にコピーする関数
    
    Args:
        src_dir (str): コピー元ディレクトリパス
        dst_dir (str): コピー先ディレクトリパス
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        while True:
            try:
                if os.path.isdir(src_path):
                    copy_directory(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                print("２秒後に再試行...")
                time.sleep(2)
def get_firing_rate(spikemon, start_time=None, end_time=None, enable_print:bool=False, mode:str="rate"):
    """
    スパイクモニターからニューロンごとの発火率を取得する
    
    Args:
        spikemon (SpikeMonitor): スパイクモニター
        start_time (float): 開始時間(ms)
        end_time (float): 終了時間(ms)
        enable_print (bool): 発火率を表示するか
        mode (str): 発火率の計算方法
            "rate": 発火率を計算
            "count": 発火回数を計算
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
    
    if mode == "rate":
        firing_rates = spike_counts / (duration / second)
    elif mode == "count":
        firing_rates = spike_counts
    
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

# ===================================== パラメータの保存・読み込みｆ ==========================================

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

def save_parameters(save_path:str, parameters:dict):
    """
    JSONファイルにパラメータを保存します。Brian2の単位変換もいます。

    Args:
        save_path (str): ディレクトリのパス
        parameters (dict): パラメータの辞書
    Returns:
        None
    """
    with open(save_path, "w") as f:
        json.dump(parameters, f, indent=4, default=convert_quantity)
        
def memo_assigned_labels(save_path, assigned_labels):
    """
    ニューロンに割り当てらたラベルを読みやす形でテキストファイルに保存します。
    """
    with open(save_path, "w") as f:
        f.write("[Assigned labels]\n")
        for i in range(len(assigned_labels)):
            f.write(f"\tneuron {i}: {assigned_labels[i]}\n")
            
def save_assigned_labels(save_path, assigned_labels):
    """
    ニューロンに割り当てられたラベルをValidate時に使用するためにpklファイルに保存します。
    """
    with open(save_path, "wb") as f:
        pkl.dump(assigned_labels, f)

# ===================================== モニターデータの保存 ==========================================
        
def save_monitor(monitor, save_path, record_variables=None, compress=True):
    """
    モニターデータを保存する関数
    
    Args:
        monitor: Brian2のSpikeMonitorまたはStateMonitor
        save_path (str): 保存先のパス
        record_variables (list, optional): StateMonitorの場合の記録変数リスト
        compress (bool): 圧縮して保存するかどうか（デフォルト：True）
    """
    with open(save_path, "wb") as f:
        if isinstance(monitor, SpikeMonitor):
            monitor_data = SpikeMonitorData(monitor)
        elif isinstance(monitor, StateMonitor):
            monitor_data = StateMonitorData(monitor, monitor.record_variables)
        elif isinstance(monitor, SpikeMonitorData):
            monitor_data = monitor
        elif isinstance(monitor, StateMonitorData):
            monitor_data = monitor
        else:
            raise ValueError("保存できるモニターはSpikeMonitor, StateMonitor, SpikeMonitorData, StateMonitorDataのみです。")
        
        if compress:
            pkl.dump(monitor_data, f, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            pkl.dump(monitor_data, f)

def load_monitor(file_path:str):
    """
    pklファイル二保存されたモニターを読み込みます。以下のような方法で実際のモニター同様にデータを読み出せます。
    
    例：
    monitor = load_monitor(file_path)
    print(monitor.name)
    print(monitor.t)
    """
    with open(file_path, "rb") as f:
        monitor = pkl.load(f)

    return monitor

# ===================================== 分割してモニターを保存 ==========================================
def save_all_monitors(save_path:str, network, index:int=None, compress=True):
    # NOTE 分割保存すると謎に恐ろしいほどメモリを消費するバグあり
    """
    ネットワーク内の全てのモニターを保存する関数。
    indexを指定すると，モニター名のフォルダを作成し，その中に分割したモニターを保存し，モニターをクリアします。
    モニターがネットワークに含まれていなかった場合は，何も行いません。
    指定されたパスが存在しなかった場合は，作成します。
    
    Args:
        save_path (str): 保存先のパス
        network (Network): モニターを含むネットワーク
        index (int, optional): 分割保存時のインデックス
        compress (bool): 圧縮して保存するかどうか
    """
    os.makedirs(save_path, exist_ok=True)
    for item in network.objects:
        if isinstance(item, SpikeMonitor):
            if index is None:
                save_monitor(item, os.path.join(save_path, f"{item.name}.pkl"), compress=compress)
            else:
                os.makedirs(os.path.join(save_path, item.name), exist_ok=True)
                save_monitor(item, os.path.join(save_path, item.name, f"{index}.pkl"), compress=compress)
                # 既存のモニターを削除
                network.remove(item)
                # 新しいモニターを作成して追加
                new_spikemon = SpikeMonitor(item.source, record=True, name=item.name)
                network.add(new_spikemon)
                
        elif isinstance(item, StateMonitor):
            if index is None:
                save_monitor(item, os.path.join(save_path, f"{item.name}.pkl"), item.record_variables, compress=compress)
            else:
                os.makedirs(os.path.join(save_path, item.name), exist_ok=True)
                save_monitor(item, os.path.join(save_path, item.name, f"{index}.pkl"), item.record_variables, compress=compress)
                # 既存のモニターを削除
                network.remove(item)
                # 新しいモニターを作成して追加
                new_statemon = StateMonitor(item.source, item.record_variables, record=True, name=item.name)
                network.add(new_statemon)

def merge_separated_monitors(dir_path:str):
    """
    分割して保存したモニターを結合して返します。
    結合したいモニターが入っているディレクトリのパスを指定します。
    """
    # dir_path内のpklファイルのパスを取得
    monitor_paths = []
    for file in os.listdir(dir_path):
        if file.endswith(".pkl"):
            monitor_paths.append(os.path.join(dir_path, file))
    monitor_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if len(monitor_paths) == 0:
        print(f"[WARNING] 指定されたディレクトリにモニターファイルが見つかりません。")
        return None
    
    monitors = [load_monitor(path) for path in monitor_paths]
    merged_monitor = monitors[0]
    for monitor in monitors[1:]:
        merged_monitor.extend(monitor)
    return merged_monitor



