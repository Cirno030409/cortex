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
import matplotlib
matplotlib.use('TkAgg')  # または 'Qt5Agg'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from brian2.units import *
from matplotlib.widgets import Button
import mpld3
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "Times New Roman"

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
        synapses (Synapses): Synapsesオブジェクト
        value: 正規化の目標値
        value2: minmaxメソッド使用時の最大値
        method: 正規化方法

    Returns:
        None
    """
    n_i = synapse.N_incoming_post[0]
    n_j = synapse.N_outgoing_pre[0]
    
    # 重みの単位を取得
    unit = synapse.w.unit
    
    # 単位を外して計算用の配列を作成
    connections = np.zeros((n_i, n_j))
    connections[synapse.i, synapse.j] = synapse.w/unit
    
    if method == "sum":
        col_sum = np.sum(connections, axis=0)
        # ゼロ除算を防ぐ
        col_sum[col_sum == 0] = 1
        col_factor = (value/unit) / col_sum
        for i in range(n_j):
            connections[:, i] *= col_factor[i]
    elif method == "sum_square":
        col_sum = np.sum(connections**2, axis=0)
        # ゼロ除算を防ぐ
        col_sum[col_sum == 0] = 1
        col_factor = np.sqrt((value/unit)**2 / col_sum)
        for i in range(n_j):
            connections[:, i] *= col_factor[i]
    elif method == "minmax":
        for i in range(n_j):
            col = connections[:, i]
            if np.max(col) > np.min(col):  # 範囲が0でない場合のみ正規化
                col_min = np.min(col)
                col_max = np.max(col)
                col_range = col_max - col_min
                connections[:, i] = (col - col_min) / col_range * ((value2 - value)/unit) + (value/unit)
    elif method == "minus":
        connections -= (value/unit)
    else:
        raise ValueError("methodには'sum', 'sum_square', 'minmax', 'minus'のいずれかを指定してください。")
    
    # 単位を付けて戻す
    synapse.w = connections[synapse.i, synapse.j] * unit
        
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

def visualize_network(network, save_path:str=None):
    """
    brian2のネットワーク構造を可視化する関数 (改善版)
    ネットワーク内のNeuronGroupの大きさに応じたノードサイズや、凡例、矢印スタイルの改善を行いました。
    シナプス強度に応じて矢印の太さが変化します。
    階層構造を考慮したレイアウトを使用します。
    シナプス名はマウスホバーで表示されます。
    ニューロン名に"M0"や"M1"などが含まれる場合、カラム別に配置します。
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import re
    from brian2.units import amp, siemens, pA, nS

    # 図の作成（先に作成して他の処理の前に配置）
    fig = plt.figure(figsize=(10, 10))  # より大きなサイズに変更
    fig.canvas.manager.set_window_title('Network Structure')  # ウィンドウタイトルを設定
    ax = fig.add_subplot(111)

    # グラフオブジェクトの作成（有向グラフ）
    G = nx.DiGraph()

    # 選択状態を管理する変数を初期化
    selected_nodes = set()  # 単一のノードから複数のノードを選択できるようにセットに変更

    # シナプス強度を保存するための辞書
    synapse_weights = {}
    synapse_types = {}  # 興奮性/抑制性の情報を保存

    # ニューロングループとシナプスの情報を収集
    for obj in network.objects:
        obj_type = str(type(obj))
        if 'NeuronGroup' in obj_type or 'LIF_Neuron' in obj_type or 'PoissonGroup' in obj_type or 'Poisson_Input_Neuron' in obj_type:
            # ノードサイズを固定し、ラベルにニューロン数を表示
            G.add_node(obj.name, type='neuron', size=3000, label=f"{obj.name}\nN={obj.N}")
        elif 'Synapses' in obj_type:
            # シナプス強度をwメンバ変数から取得
            if hasattr(obj, 'w'):
                w = obj.w
                # Quantityオブジェクトの場合は数値に変換
                if hasattr(w, 'dim'):
                    if w.dim == amp.dim:  # 電流の場合
                        # 配列の平均値を使用
                        weight = float(np.mean(w)/pA)  # pAを単位として使用
                        weight_unit = 'pA'
                    elif w.dim == siemens.dim:  # コンダクタンスの場合
                        # 配列の平均値を使用
                        weight = float(np.mean(w)/nS)  # nSを単位として使用
                        weight_unit = 'nS'
                    else:
                        # 配列の平均値を使用
                        weight = float(np.mean(w))  # その他の場合は単純に数値化
                        weight_unit = ''
                else:
                    # 配列の平均値を使用
                    weight = float(np.mean(w))
                    weight_unit = ''
            else:
                weight = 1.0  # デフォルト値
                weight_unit = ''
            
            # シナプスタイプを取得
            synapse_type = getattr(obj, 'exc_or_inh', 'exc')  # デフォルトは興奮性
            
            G.add_edge(obj.source.name, obj.target.name, name=obj.name, weight=weight, weight_unit=weight_unit, synapse_type=synapse_type)
            synapse_weights[(obj.source.name, obj.target.name)] = weight
            synapse_types[(obj.source.name, obj.target.name)] = synapse_type

    if len(G.nodes()) == 0:
        print("警告: ネットワークにニューロングループが見つかりません。")
        return

    # カスタムレイアウトの作成（階層構造を考慮）
    pos = {}
    layer_spacing = 3.0  # 層間の垂直距離を2.0から3.0に増加
    node_spacing = 2.0   # 同じ層内でのノード間隔を1.0から2.0に増加

    # レイヤーごとのノードを分類
    layer_nodes = {
        'L23': [],
        'L4': [],
        'L5': [],
        'L6': [],
        'noise': []
    }

    # カラム情報を格納する辞書
    column_info = {}

    # ニューロン名からカラム情報を抽出する正規表現パターン
    # 複数のパターンを試す
    column_patterns = [
        re.compile(r'M(\d+)'),           # M0, M1 などの標準パターン
        re.compile(r'minicolumn(\d+)'),  # minicolumn0, minicolumn1 などのパターン
        re.compile(r'column[_-]?(\d+)'), # column0, column_0, column-0 などのパターン
        re.compile(r'col[_-]?(\d+)')     # col0, col_0, col-0 などのパターン
    ]

    # 各ノードのカラム情報を取得
    for node in G.nodes():
        for pattern in column_patterns:
            match = pattern.search(str(node))
            if match:
                column_number = int(match.group(1))
                column_info[node] = column_number
                break  # 最初にマッチしたパターンを使用

    # カラムの数を確認
    column_numbers = set(column_info.values()) if column_info else set()
    
    # # カラムが見つかった場合は出力
    # if column_numbers:
    #     print(f"検出されたカラム: {sorted(column_numbers)}")
    #     print(f"カラム情報を持つノード数: {len(column_info)}")
    #     for node, col in column_info.items():
    #         print(f"  ノード: {node}, カラム: {col}")

    # 各レイヤーのノードに座標を割り当て
    y_positions = {
        'L23': 6 * layer_spacing,  # より広い垂直間隔
        'L4': 4.5 * layer_spacing,
        'L5': 3 * layer_spacing,
        'L6': 1.5 * layer_spacing,
        'noise': 0
    }

    # 未分類ノードを保存するリスト
    other_nodes = []

    for node in G.nodes():
        if 'L23' in node:
            layer_nodes['L23'].append(node)
        elif 'L4' in node:
            layer_nodes['L4'].append(node)
        elif 'L5' in node:
            layer_nodes['L5'].append(node)
        elif 'L6' in node:
            layer_nodes['L6'].append(node)
        elif 'noise' in node.lower():
            layer_nodes['noise'].append(node)
        else:
            other_nodes.append(node)

    # カラム情報がある場合、各レイヤー内でさらにカラムごとに分類
    if column_numbers:
        column_count = len(column_numbers)
        column_width = 3.0  # カラム間の水平距離
        
        for layer, nodes in layer_nodes.items():
            if not nodes:
                continue
            
            # レイヤー内のノードをカラムごとに分類
            column_nodes = {col: [] for col in column_numbers}
            non_column_nodes = []
            
            for node in nodes:
                if node in column_info:
                    column_nodes[column_info[node]].append(node)
                else:
                    non_column_nodes.append(node)
            
            # カラムごとにノードの座標を割り当て
            y = y_positions[layer]
            
            # 非カラムノードがある場合、左側に配置
            if non_column_nodes:
                x_offset = -(column_count * column_width / 2) - node_spacing
                for i, node in enumerate(sorted(non_column_nodes)):
                    x = x_offset - i * node_spacing
                    pos[node] = np.array([x, y])
            
            # カラムノードを配置
            for col_idx, col in enumerate(sorted(column_numbers)):
                col_nodes = column_nodes[col]
                if not col_nodes:
                    continue
                
                x_center = -column_count * column_width / 2 + col_idx * column_width + column_width / 2
                
                # カラム内でのx座標を計算
                local_x_offset = -(len(col_nodes) - 1) * node_spacing / 2
                for i, node in enumerate(sorted(col_nodes)):
                    x = x_center + local_x_offset + i * node_spacing
                    pos[node] = np.array([x, y])
    else:
        # カラム情報がない場合は元の配置方法を使用
        for layer, nodes in layer_nodes.items():
            if not nodes:
                continue
            
            # 各レイヤー内でのx座標を計算
            x_offset = -(len(nodes) - 1) * node_spacing / 2
            for i, node in enumerate(sorted(nodes)):
                x = x_offset + i * node_spacing
                y = y_positions[layer]
                pos[node] = np.array([x, y])

    # 未分類ノードがある場合、Spring Layoutで配置
    if other_nodes:
        # 未分類ノードのサブグラフを作成
        other_subgraph = G.subgraph(other_nodes)
        # Spring Layoutで配置を計算（y座標を1.5 * layer_spacingに制限）
        other_pos = nx.spring_layout(other_subgraph, k=2.0, iterations=50, seed=42)  # kパラメータを1.0から2.0に増加
        
        # y座標を調整（1.5 * layer_spacingの位置に固定）
        target_y = 1.5 * layer_spacing
        y_min = min(p[1] for p in other_pos.values())
        y_max = max(p[1] for p in other_pos.values())
        y_scale = 1.0  # y方向の広がりを0.5から1.0に増加
        
        for node, (x, y) in other_pos.items():
            # y座標を正規化して配置
            normalized_y = (y - y_min) / (y_max - y_min if y_max != y_min else 1)
            new_y = target_y + (normalized_y - 0.5) * y_scale
            pos[node] = np.array([x * 3, new_y])  # x座標の倍率を2から3に増加

    # エッジの描画（ノードの前に描画）
    if G.edges():
        # 重みが0でないエッジのみを抽出
        non_zero_edges = []
        for u, v in G.edges():
            # シナプスオブジェクトを探す
            synapse_obj = None
            for obj in network.objects:
                if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                    synapse_obj = obj
                    break
            
            if synapse_obj is None:
                continue

            # 接続確率を取得
            try:
                p_value = synapse_obj.p
            except AttributeError:
                print(f"[WARNING] シナプス {G[u][v]['name']} にp属性が存在しません。")
                continue

            # 重みと接続確率の両方が0より大きいエッジのみを追加
            if abs(G[u][v]['weight']) > 0 and p_value > 0:
                non_zero_edges.append((u, v))

        if not non_zero_edges:  # 有効なエッジがない場合は処理をスキップ
            return

        # 重みが0でないエッジの重みのみを使用して正規化
        weights = [abs(G[u][v]['weight']) for u, v in non_zero_edges]
        max_weight = max(weights)
        min_weight = min(weights)

        if max_weight == min_weight:
            normalized_weights = [2.0] * len(weights)
        else:
            normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 4 + 1 for w in weights]

        # エッジの色をシナプスタイプに応じて設定（重みが0でないエッジのみ）
        edge_colors = ['red' if G[u][v]['synapse_type'] == 'exc' else 'blue' for u, v in non_zero_edges]

        # 自己結合と通常の結合を分離（重みが0でないエッジのみ）
        self_edges = [(u, v) for (u, v) in non_zero_edges if u == v]
        normal_edges = [(u, v) for (u, v) in non_zero_edges if u != v]

        # 通常の結合を重みの降順でソート
        if normal_edges:
            normal_edges_with_weights = [(u, v, normalized_weights[i]) 
                                           for i, (u, v) in enumerate(non_zero_edges) if u != v]
            normal_edges_with_weights.sort(key=lambda x: x[2], reverse=True)
            
            # 重みの大きい順に描画（太い線が下になるように）
            for u, v, weight in normal_edges_with_weights:
                edge_idx = non_zero_edges.index((u, v))
                # シナプスオブジェクトを探す
                synapse_obj = None
                for obj in network.objects:
                    if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                        synapse_obj = obj
                        break
                
                # 基本の透明度を接続確率に基づいて設定
                if synapse_obj is not None:
                    try:
                        p_value = synapse_obj.p
                        base_alpha = 0.2 + 0.6 * p_value  # p=0で0.2, p=1で0.8
                    except AttributeError:
                        base_alpha = 0.5  # デフォルト値
                else:
                    base_alpha = 0.5

                # 選択状態に応じて透明度を調整
                if selected_nodes:
                    if len(selected_nodes) == 1:
                        edge_alpha = base_alpha if list(selected_nodes)[0] in [u, v] else 0.0  # 0.1から0.0に変更
                    else:
                        edge_alpha = base_alpha if u in selected_nodes and v in selected_nodes else 0.0  # 0.1から0.0に変更
                else:
                    edge_alpha = base_alpha

                nx.draw_networkx_edges(G, pos,
                                   edgelist=[(u, v)],
                                   edge_color=[edge_colors[edge_idx]],
                                   arrows=True,
                                   arrowsize=20,
                                   width=weight,
                                   arrowstyle='-|>',
                                   node_size=3000,
                                   alpha=edge_alpha)

        # 自己結合を重みの降順でソート
        if self_edges:
            self_edges_with_weights = [(u, v, normalized_weights[i]) 
                                         for i, (u, v) in enumerate(non_zero_edges) if u == v]
            self_edges_with_weights.sort(key=lambda x: x[2], reverse=True)
            
            # 重みの大きい順に描画
            for u, v, weight in self_edges_with_weights:
                edge_idx = non_zero_edges.index((u, v))
                # シナプスオブジェクトを探す
                synapse_obj = None
                for obj in network.objects:
                    if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                        synapse_obj = obj
                        break
                
                # 基本の透明度を接続確率に基づいて設定
                if synapse_obj is not None:
                    try:
                        p_value = synapse_obj.p
                        base_alpha = 0.2 + 0.6 * p_value  # p=0で0.2, p=1で0.8
                    except AttributeError:
                        base_alpha = 0.5  # デフォルト値
                else:
                    base_alpha = 0.5

                # 選択状態に応じて透明度を調整
                if selected_nodes:
                    if len(selected_nodes) == 1:
                        edge_alpha = base_alpha if list(selected_nodes)[0] == u else 0.0  # 0.1から0.0に変更
                    else:
                        edge_alpha = base_alpha if u in selected_nodes else 0.0  # 0.1から0.0に変更
                else:
                    edge_alpha = base_alpha

                nx.draw_networkx_edges(G, pos,
                                   edgelist=[(u, v)],
                                   edge_color=[edge_colors[edge_idx]],
                                   arrows=True,
                                   arrowsize=15,
                                   width=weight,
                                   arrowstyle='-|>',
                                   node_size=3000,
                                   connectionstyle='arc3, rad=0.15',
                                   alpha=edge_alpha)

        # エッジラベルの辞書を作成（ホバー表示用）（重みが0でないエッジのみ）
        edge_labels = {}
        for u, v in non_zero_edges:
            # シナプスオブジェクトを探す
            synapse_obj = None
            for obj in network.objects:
                if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                    synapse_obj = obj
                    break
            
            if synapse_obj is None:
                print(f"[WARNING] シナプス {G[u][v]['name']} のオブジェクトが見つかりません。")
                continue
                
            # 接続確率を取得（必ず存在するはず）
            try:
                p_value = synapse_obj.p
                if p_value is None:
                    print(f"[WARNING] シナプス {synapse_obj.name} のp属性がNoneです。")
            except AttributeError:
                print(f"[WARNING] シナプス {synapse_obj.name} にp属性が存在しません。")
                print(f"[DEBUG] シナプスの属性一覧: {dir(synapse_obj)}")
                p_value = None
            
            p_str = f", p={p_value:.3f}" if p_value is not None else ""
            
            # 時定数を取得
            tau_value = None
            if 'tau' in synapse_obj.namespace:
                tau = synapse_obj.namespace['tau']
                if hasattr(tau, 'dim'):
                    tau_value = float(tau/ms)  # msを単位として使用
                else:
                    tau_value = float(tau)
            tau_str = f", τ={tau_value:.1f} ms" if tau_value is not None else ""
            
            # weighting_factorがある場合、デフォルト値(1.0)と異なる場合のみ表示
            wf_str = ""
            if hasattr(synapse_obj, 'weighting_factor'):
                wf_value = synapse_obj.weighting_factor
                # デフォルト値であっても常に表示する
                wf_str = f", wf={wf_value:.3f}"
            
            edge_labels[(u, v)] = f"{G[u][v]['name']}\n(w={G[u][v]['weight']:.1f} {G[u][v]['weight_unit']}{p_str}{tau_str}{wf_str})"

        # ホバーイベントの設定
        def hover(event):
            if event.inaxes != ax:
                return
            
            # マウス位置を取得
            mouse_x, mouse_y = event.xdata, event.ydata
            if mouse_x is None or mouse_y is None:
                return

            # 最も近いエッジを探す
            min_dist = float('inf')
            close_edges = []  # 近くのエッジをすべて保存
            
            # 選択されたノードに関連するエッジのみを対象とする
            target_edges = []
            if selected_nodes:
                if len(selected_nodes) == 1:  # 1つのノードだけ選択されている場合
                    # 選択されたノードに接続している全てのエッジを対象とする
                    target_edges = [(u, v) for (u, v) in G.edges() if list(selected_nodes)[0] in [u, v]]
                else:  # 複数のノードが選択されている場合
                    # 選択されたノード間のエッジのみを抽出
                    target_edges = [(u, v) for (u, v) in G.edges() if u in selected_nodes and v in selected_nodes]
            else:
                # 選択されていない場合は全てのエッジを対象とする
                target_edges = list(G.edges())

            for (u, v) in target_edges:
                # エッジの始点と終点の座標（node_positionsを使用）
                start = node_positions[u]
                end = node_positions[v]
                
                if u == v:  # 自己結合の場合
                    # 自己結合の円弧の中心と半径を計算
                    center_x = start[0]
                    center_y = start[1] + 0.05  # オフセットを0.05に縮小
                    radius = 0.1  # 半径を0.15から0.1に縮小
                    
                    # マウス位置と円弧の中心との距離を計算
                    dist_to_center = np.sqrt((mouse_x - center_x)**2 + (mouse_y - center_y)**2)
                    
                    # 円弧からの距離の許容範囲を0.1から0.05に縮小
                    if abs(dist_to_center - radius) < 0.05:
                        close_edges.append((u, v, abs(dist_to_center - radius)))
                    continue
                
                # 通常の接続の場合（既存のコード）
                px = end[0] - start[0]
                py = end[1] - start[1]
                norm = px * px + py * py
                if norm == 0:
                    continue
                    
                t = max(0, min(1, ((mouse_x - start[0]) * px + (mouse_y - start[1]) * py) / norm))
                
                point_x = start[0] + t * px
                point_y = start[1] + t * py
                
                dist = np.sqrt((mouse_x - point_x)**2 + (mouse_y - point_y)**2)
                
                # 検出範囲を0.1から0.05に縮小
                if 0 <= t <= 1 and dist < 0.05:  # 検出範囲を半分に縮小
                    edge_vector = np.array([px, py])
                    mouse_vector = np.array([mouse_x - start[0], mouse_y - start[1]])
                    
                    if np.all(edge_vector == 0) or np.all(mouse_vector == 0):
                        continue
                    
                    cos_angle = np.dot(edge_vector, mouse_vector) / (np.linalg.norm(edge_vector) * np.linalg.norm(mouse_vector))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    
                    # 角度の許容範囲をより厳密に（π/6からπ/8に変更）
                    if angle < np.pi/8 or angle > 7*np.pi/8:
                        close_edges.append((u, v, dist))

            # 既存のアノテーションを削除
            for artist in ax.texts:
                if hasattr(artist, 'is_edge_label'):
                    artist.remove()

            # 近くのエッジが見つかった場合、ラベルを表示
            if close_edges:
                # 距離でソート
                close_edges.sort(key=lambda x: x[2])
                
                # ラベルテキストを作成
                label_texts = []
                for u, v, _ in close_edges:
                    if (u, v) in edge_labels:
                        # 矢印の方向を示す文字列を作成
                        if u == v:  # 自己結合の場合
                            arrow = "⟲"  # 循環を示す矢印
                        else:  # 通常の接続の場合
                            dx = node_positions[v][0] - node_positions[u][0]  # x方向の差分
                            dy = node_positions[v][1] - node_positions[u][1]  # y方向の差分
                            
                            # 斜め方向を含む8方向の判定（閾値を調整）
                            dx_abs = abs(dx)
                            dy_abs = abs(dy)
                            
                            if dx_abs < 0.3 * dy_abs:  # x方向の差が小さい場合は垂直
                                arrow = "↑" if dy > 0 else "↓"
                            elif dy_abs < 0.3 * dx_abs:  # y方向の差が小さい場合は水平
                                arrow = "→" if dx > 0 else "←"
                            else:  # 斜め
                                if dx > 0:  # 右向き
                                    arrow = "↗" if dy > 0 else "↘"
                                else:  # 左向き
                                    arrow = "↖" if dy > 0 else "↙"
                        
                        # シナプスタイプに応じてマーカーを設定
                        marker = "+" if G[u][v]['synapse_type'] == 'exc' else "-"
                        # edge_labelsから完全なラベル情報を取得
                        label_text = f"{arrow} [{marker}] {edge_labels[(u, v)]}"
                        label_texts.append(label_text)

                # すべてのラベルを結合
                combined_label = "\n".join(label_texts)
                
                # 表示位置を計算（最も近いエッジの中点または自己結合の上部）
                closest_u, closest_v, _ = close_edges[0]
                if closest_u == closest_v:  # 自己結合の場合
                    mid_x = node_positions[closest_u][0]
                    mid_y = node_positions[closest_u][1] + 0.2  # ラベル位置をさらに下げる
                else:  # 通常の接続の場合
                    mid_x = (node_positions[closest_u][0] + node_positions[closest_v][0]) / 2
                    mid_y = (node_positions[closest_u][1] + node_positions[closest_v][1]) / 2
                
                # ラベルを表示
                annotation = ax.text(mid_x, mid_y + 0.2, combined_label,
                                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.9,
                                           pad=5),
                                   ha='center', va='bottom',
                                   fontsize=13,
                                   fontfamily='DejaVu Sans',  # フォントを追加
                                   zorder=100)
                annotation.is_edge_label = True
                fig.canvas.draw_idle()

        # マウスのモーションイベントを接続
        fig.canvas.mpl_connect('motion_notify_event', hover)

    # ニューロンノードの描画
    neuron_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'neuron']
    if neuron_nodes:
        # カラム情報がある場合、カラムごとに異なる色を使用
        if column_numbers:
            # カラム別のカラーマップを作成
            import matplotlib.cm as cm
            colormap = cm.get_cmap('tab10', max(len(column_numbers), 1))  # カラム数に応じたカラーマップ
            column_colors = {}
            column_nodes = {}
            
            # カラムごとに色を割り当て
            for i, col in enumerate(sorted(column_numbers)):
                column_colors[col] = colormap(i)
                column_nodes[col] = [n for n in neuron_nodes if n in column_info and column_info[n] == col]
            
            # カラム別にノードを描画
            for col in sorted(column_numbers):
                col_nodes = column_nodes[col]
                if col_nodes:
                    nx.draw_networkx_nodes(G, pos,
                                       nodelist=col_nodes,
                                       node_color=[column_colors[col]] * len(col_nodes),
                                       node_size=3000,
                                       edgecolors='black',
                                       linewidths=2)
            
            # カラム情報のないノードを描画
            non_col_nodes = [n for n in neuron_nodes if n not in column_info]
            if non_col_nodes:
                nx.draw_networkx_nodes(G, pos,
                                   nodelist=non_col_nodes,
                                   node_color='skyblue',
                                   node_size=3000,
                                   edgecolors='black',
                                   linewidths=2)
        else:
            # カラム情報がない場合は全て同じ色で描画
            node_collection = nx.draw_networkx_nodes(G, pos,
                                nodelist=neuron_nodes,
                                node_color='skyblue',
                                node_size=3000,
                                edgecolors='black',
                                linewidths=2)

        # ノードラベルの描画
        labels = {n: G.nodes[n]['label'] for n in neuron_nodes}
        label_artists = nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        # ドラッグ機能の実装
        dragged_node = [None]
        selected_nodes = set()  # 選択されたノードを保存
        node_positions = pos.copy()  # 現在のノード位置をコピー
        original_positions = pos.copy()  # 元の位置を保存
        mouse_moved = [False]  # マウスが移動したかどうかを追跡
        initial_click_pos = [None]  # クリック時の初期位置を保存
        
        def on_press(event):
            if event.inaxes != ax:
                return
            # マウス位置を取得
            mouse_x, mouse_y = event.xdata, event.ydata
            if mouse_x is None or mouse_y is None:
                return
            
            # 最も近いノードを探す
            closest_node = None
            min_dist = float('inf')
            for node in neuron_nodes:
                node_pos = node_positions[node]
                dist = np.sqrt((mouse_x - node_pos[0])**2 + (mouse_y - node_pos[1])**2)
                if dist < min_dist and dist < 0.5:  # ノードの近くにある場合のみ
                    min_dist = dist
                    closest_node = node

            if closest_node is None:  # ノードが見つからない場合は何もしない
                return

            if event.button == 1:  # 左クリック
                # マウス移動フラグをリセット
                mouse_moved[0] = False
                initial_click_pos[0] = (mouse_x, mouse_y)
                dragged_node[0] = closest_node
        
        def on_motion(event):
            if event.inaxes != ax:
                dragged_node[0] = None  # 追加：範囲外に出た場合はドラッグ状態をリセット
                mouse_moved[0] = False
                initial_click_pos[0] = None
                return
            
            if dragged_node[0] is None:
                return
            
            # マウスが十分に移動したかチェック
            if initial_click_pos[0] is not None:
                dx = event.xdata - initial_click_pos[0][0]
                dy = event.ydata - initial_click_pos[0][1]
                if np.sqrt(dx**2 + dy**2) > 0.1:  # 閾値を設定
                    mouse_moved[0] = True
            
            # マウスボタンの状態をチェック
            if not event.button:  # マウスボタンが押されていない場合
                dragged_node[0] = None
                mouse_moved[0] = False
                initial_click_pos[0] = None
                return
            
            # ノードの位置を更新
            node_positions[dragged_node[0]] = np.array([event.xdata, event.ydata])
            
            # 再描画
            ax.clear()
            plt.axis('off')
            
            # エッジの再描画
            if G.edges():
                # シナプス強度を正規化して線の太さに変換
                weights = [abs(G[u][v]['weight']) for u, v in G.edges()]

                # 重みが0でないエッジのみを抽出
                non_zero_edges = []
                for u, v in G.edges():
                    # シナプスオブジェクトを探す
                    synapse_obj = None
                    for obj in network.objects:
                        if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                            synapse_obj = obj
                            break
                    
                    if synapse_obj is None:
                        continue

                    # 接続確率を取得
                    try:
                        p_value = synapse_obj.p
                    except AttributeError:
                        print(f"[WARNING] シナプス {G[u][v]['name']} にp属性が存在しません。")
                        continue

                    # 重みと接続確率の両方が0より大きいエッジのみを追加
                    if abs(G[u][v]['weight']) > 0 and p_value > 0:
                        non_zero_edges.append((u, v))

                if not non_zero_edges:  # 有効なエッジがない場合は処理をスキップ
                    return

                # 重みが0でないエッジの重みのみを使用して正規化
                weights = [abs(G[u][v]['weight']) for u, v in non_zero_edges]
                max_weight = max(weights)
                min_weight = min(weights)

                if max_weight == min_weight:
                    normalized_weights = [2.0] * len(weights)
                else:
                    normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 4 + 1 for w in weights]

                # エッジの色をシナプスタイプに応じて設定（重みが0でないエッジのみ）
                edge_colors = ['red' if G[u][v]['synapse_type'] == 'exc' else 'blue' for u, v in non_zero_edges]

                # 自己結合と通常の結合を分離（重みが0でないエッジのみ）
                self_edges = [(u, v) for (u, v) in non_zero_edges if u == v]
                normal_edges = [(u, v) for (u, v) in non_zero_edges if u != v]

                # 通常の結合を重みの降順でソート
                if normal_edges:
                    normal_edges_with_weights = [(u, v, normalized_weights[i]) 
                                           for i, (u, v) in enumerate(non_zero_edges) if u != v]
                    normal_edges_with_weights.sort(key=lambda x: x[2], reverse=True)
                    
                    # 重みの大きい順に描画（太い線が下になるように）
                    for u, v, weight in normal_edges_with_weights:
                        edge_idx = non_zero_edges.index((u, v))
                        # シナプスオブジェクトを探す
                        synapse_obj = None
                        for obj in network.objects:
                            if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                                synapse_obj = obj
                                break
                        
                        # 基本の透明度を接続確率に基づいて設定
                        if synapse_obj is not None:
                            try:
                                p_value = synapse_obj.p
                                base_alpha = 0.2 + 0.6 * p_value  # p=0で0.2, p=1で0.8
                            except AttributeError:
                                base_alpha = 0.5  # デフォルト値
                        else:
                            base_alpha = 0.5

                        # 選択状態に応じて透明度を調整
                        if selected_nodes:
                            if len(selected_nodes) == 1:
                                edge_alpha = base_alpha if list(selected_nodes)[0] in [u, v] else 0.0  # 0.1から0.0に変更
                            else:
                                edge_alpha = base_alpha if u in selected_nodes and v in selected_nodes else 0.0  # 0.1から0.0に変更
                        else:
                            edge_alpha = base_alpha

                        nx.draw_networkx_edges(G, node_positions,  # posからnode_positionsに変更
                                           edgelist=[(u, v)],
                                           edge_color=[edge_colors[edge_idx]],
                           arrows=True,
                                           arrowsize=20,
                                           width=weight,
                           arrowstyle='-|>',
                                           node_size=3000,
                                           alpha=edge_alpha)

                # 自己結合を重みの降順でソート
                if self_edges:
                    self_edges_with_weights = [(u, v, normalized_weights[i]) 
                                             for i, (u, v) in enumerate(non_zero_edges) if u == v]
                    self_edges_with_weights.sort(key=lambda x: x[2], reverse=True)
                    
                    # 重みの大きい順に描画
                    for u, v, weight in self_edges_with_weights:
                        edge_idx = non_zero_edges.index((u, v))
                        # シナプスオブジェクトを探す
                        synapse_obj = None
                        for obj in network.objects:
                            if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                                synapse_obj = obj
                                break
                        
                        # 基本の透明度を接続確率に基づいて設定
                        if synapse_obj is not None:
                            try:
                                p_value = synapse_obj.p
                                base_alpha = 0.2 + 0.6 * p_value  # p=0で0.2, p=1で0.8
                            except AttributeError:
                                base_alpha = 0.5  # デフォルト値
                        else:
                            base_alpha = 0.5

                        # 選択状態に応じて透明度を調整
                        if selected_nodes:
                            if len(selected_nodes) == 1:
                                edge_alpha = base_alpha if list(selected_nodes)[0] == u else 0.0  # 0.1から0.0に変更
                            else:
                                edge_alpha = base_alpha if u in selected_nodes else 0.0  # 0.1から0.0に変更
                        else:
                            edge_alpha = base_alpha

                        nx.draw_networkx_edges(G, node_positions,  # posからnode_positionsに変更
                                           edgelist=[(u, v)],
                                           edge_color=[edge_colors[edge_idx]],
                                           arrows=True,
                                           arrowsize=15,
                                           width=weight,
                                           arrowstyle='-|>',
                                           node_size=3000,
                                           connectionstyle='arc3, rad=0.15',
                                           alpha=edge_alpha)
            
            # ノードの再描画
            nx.draw_networkx_nodes(G, node_positions,
                               nodelist=neuron_nodes,
                               node_color='skyblue',
                               node_size=3000,
                               edgecolors='black',
                               linewidths=2)
            
            # ラベルの再描画
            nx.draw_networkx_labels(G, node_positions, labels=labels, font_size=10)
            
            # タイトルと凡例の再描画
            plt.title("Network Structure", fontsize=24, pad=20, fontweight='bold', fontfamily='Times New Roman')
            
            # 凡例項目のリスト
            legend_handles = [mpatches.Patch(color='skyblue', label='Neurons')]
            
            # カラム情報がある場合は追加
            if column_numbers:
                import matplotlib.cm as cm
                colormap = cm.get_cmap('tab10', max(len(column_numbers), 1))
                for i, col in enumerate(sorted(column_numbers)):
                    legend_handles.append(mpatches.Patch(color=colormap(i), label=f'Column {col}'))
            
            # シナプスの凡例
            legend_handles.append(mpatches.Patch(color='red', label='Excitatory Synapses\n(Line width: Weight\nOpacity: Connection probability)'))
            legend_handles.append(mpatches.Patch(color='blue', label='Inhibitory Synapses\n(Line width: Weight\nOpacity: Connection probability)'))
            
            plt.legend(handles=legend_handles, fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.axis('off')
            plt.tight_layout()
            
            # ファイルに保存
            if save_path is not None:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        def on_release(event):
            if event.inaxes != ax:
                return
            
            # マウス位置を取得
            mouse_x, mouse_y = event.xdata, event.ydata
            if mouse_x is None or mouse_y is None:
                return
            
            # 最も近いノードを探す
            closest_node = None
            min_dist = float('inf')
            for node in neuron_nodes:
                node_pos = node_positions[node]
                dist = np.sqrt((mouse_x - node_pos[0])**2 + (mouse_y - node_pos[1])**2)
                if dist < min_dist and dist < 0.5:  # ノードの近くにある場合のみ
                    min_dist = dist
                    closest_node = node

            if closest_node is None:  # ノードが見つからない場合は何もしない
                return

            if event.button == 1 and dragged_node[0] is not None:  # 左クリックの処理
                if not mouse_moved[0]:  # マウスが移動していない場合は選択操作として扱う
                    # 選択状態を切り替え
                    if closest_node in selected_nodes:
                        selected_nodes.remove(closest_node)  # 同じノードをクリックした場合は選択解除
                    else:
                        selected_nodes.add(closest_node)  # 新しいノードを選択に追加
                    
                    # 選択状態に基づいて再描画
                    ax.clear()
                    plt.axis('off')
                    
                    # エッジの再描画（選択状態を反映）
                    if G.edges():
                        # シナプス強度を正規化して線の太さに変換
                        weights = [abs(G[u][v]['weight']) for u, v in G.edges()]

                        # 重みが0でないエッジのみを抽出
                        non_zero_edges = []
                        for u, v in G.edges():
                            # シナプスオブジェクトを探す
                            synapse_obj = None
                            for obj in network.objects:
                                if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                                    synapse_obj = obj
                                    break
                            
                            if synapse_obj is None:
                                continue

                            # 接続確率を取得
                            try:
                                p_value = synapse_obj.p
                            except AttributeError:
                                print(f"[WARNING] シナプス {G[u][v]['name']} にp属性が存在しません。")
                                continue

                            # 重みと接続確率の両方が0より大きいエッジのみを追加
                            if abs(G[u][v]['weight']) > 0 and p_value > 0:
                                non_zero_edges.append((u, v))

                        if not non_zero_edges:  # 有効なエッジがない場合は処理をスキップ
                            return

                        # 重みが0でないエッジの重みのみを使用して正規化
                        weights = [abs(G[u][v]['weight']) for u, v in non_zero_edges]
                        max_weight = max(weights)
                        min_weight = min(weights)

                        if max_weight == min_weight:
                            normalized_weights = [2.0] * len(weights)
                        else:
                            normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 4 + 1 for w in weights]

                        # エッジの色をシナプスタイプに応じて設定（重みが0でないエッジのみ）
                        edge_colors = ['red' if G[u][v]['synapse_type'] == 'exc' else 'blue' for u, v in non_zero_edges]

                        # 自己結合と通常の結合を分離（重みが0でないエッジのみ）
                        self_edges = [(u, v) for (u, v) in non_zero_edges if u == v]
                        normal_edges = [(u, v) for (u, v) in non_zero_edges if u != v]

                        # 通常の結合を重みの降順でソート
                        if normal_edges:
                            normal_edges_with_weights = [(u, v, normalized_weights[i]) 
                                                   for i, (u, v) in enumerate(non_zero_edges) if u != v]
                            normal_edges_with_weights.sort(key=lambda x: x[2], reverse=True)
                            
                            # 重みの大きい順に描画（太い線が下になるように）
                            for u, v, weight in normal_edges_with_weights:
                                edge_idx = non_zero_edges.index((u, v))
                                # シナプスオブジェクトを探す
                                synapse_obj = None
                                for obj in network.objects:
                                    if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                                        synapse_obj = obj
                                        break
                                
                                # 基本の透明度を接続確率に基づいて設定
                                if synapse_obj is not None:
                                    try:
                                        p_value = synapse_obj.p
                                        base_alpha = 0.2 + 0.6 * p_value  # p=0で0.2, p=1で0.8
                                    except AttributeError:
                                        base_alpha = 0.5  # デフォルト値
                                else:
                                    base_alpha = 0.5

                                # 選択状態に応じて透明度を調整
                                if selected_nodes:
                                    if len(selected_nodes) == 1:
                                        edge_alpha = base_alpha if list(selected_nodes)[0] in [u, v] else 0.0  # 0.1から0.0に変更
                                    else:
                                        edge_alpha = base_alpha if u in selected_nodes and v in selected_nodes else 0.0  # 0.1から0.0に変更
                                else:
                                    edge_alpha = base_alpha

                                nx.draw_networkx_edges(G, node_positions,  # posからnode_positionsに変更
                                                   edgelist=[(u, v)],
                                                   edge_color=[edge_colors[edge_idx]],
                                                   arrows=True,
                           arrowsize=20,
                                                   width=weight,
                                                   arrowstyle='-|>',
                                                   node_size=3000,
                                                   alpha=edge_alpha)

                        # 自己結合を重みの降順でソート
                        if self_edges:
                            self_edges_with_weights = [(u, v, normalized_weights[i]) 
                                                     for i, (u, v) in enumerate(non_zero_edges) if u == v]
                            self_edges_with_weights.sort(key=lambda x: x[2], reverse=True)
                            
                            # 重みの大きい順に描画
                            for u, v, weight in self_edges_with_weights:
                                edge_idx = non_zero_edges.index((u, v))
                                # シナプスオブジェクトを探す
                                synapse_obj = None
                                for obj in network.objects:
                                    if isinstance(obj, Synapses) and obj.name == G[u][v]['name']:
                                        synapse_obj = obj
                                        break
                                
                                
                                # 基本の透明度を接続確率に基づいて設定
                                if synapse_obj is not None:
                                    try:
                                        p_value = synapse_obj.p
                                        base_alpha = 0.2 + 0.6 * p_value  # p=0で0.2, p=1で0.8
                                    except AttributeError:
                                        base_alpha = 0.5  # デフォルト値
                                else:
                                    base_alpha = 0.5

                                # 選択状態に応じて透明度を調整
                                if selected_nodes:
                                    if len(selected_nodes) == 1:
                                        edge_alpha = base_alpha if list(selected_nodes)[0] == u else 0.0  # 0.1から0.0に変更
                                    else:
                                        edge_alpha = base_alpha if u in selected_nodes else 0.0  # 0.1から0.0に変更
                                else:
                                    edge_alpha = base_alpha

                                nx.draw_networkx_edges(G, node_positions,  # posからnode_positionsに変更
                                                   edgelist=[(u, v)],
                                                   edge_color=[edge_colors[edge_idx]],
                                                   arrows=True,
                                                   arrowsize=15,
                                                   width=weight,
                                                   arrowstyle='-|>',
                                                   node_size=3000,
                                                   connectionstyle='arc3, rad=0.15',
                                                   alpha=edge_alpha)

                    # 関連するノードを取得
                    related_nodes = set()
                    if selected_nodes:
                        related_nodes.update(selected_nodes)
                        for u, v in G.edges():
                            if u in selected_nodes and v in selected_nodes:
                                related_nodes.add(u)
                                related_nodes.add(v)
                    
                    # ノードの再描画（関連するノードは強調表示）
                    for node in neuron_nodes:
                        node_alpha = 1.0  # デフォルトの透明度
                        if selected_nodes:
                            # 選択されたノードと関連するノードのみ強調表示
                            node_alpha = 0.9 if node in selected_nodes else 0.2
                        nx.draw_networkx_nodes(G, node_positions,
                                           nodelist=[node],
                                           node_color='skyblue',
                                           node_size=3000,
                                           edgecolors='black',
                                           linewidths=2,
                                           alpha=node_alpha)
                    
                    # ラベルの再描画（関連するノードは強調表示）
                    for node, label in labels.items():
                        label_alpha = 1.0  # デフォルトの透明度
                        if selected_nodes:
                            # 選択されたノードと関連するノードのみ強調表示
                            label_alpha = 1.0 if node in selected_nodes else 0.2
                        ax.text(node_positions[node][0], node_positions[node][1], label,
                               horizontalalignment='center', verticalalignment='center',
                               fontsize=10, alpha=label_alpha)
                    
                    # タイトルと凡例の再描画
                    plt.title("Network Structure", fontsize=24, pad=20, fontweight='bold', fontfamily='Times New Roman')
                    
                    # 凡例項目のリスト
                    legend_handles = [mpatches.Patch(color='skyblue', label='Neurons')]
                    
                    # カラム情報がある場合は追加
                    if column_numbers:
                        import matplotlib.cm as cm
                        colormap = cm.get_cmap('tab10', max(len(column_numbers), 1))
                        for i, col in enumerate(sorted(column_numbers)):
                            legend_handles.append(mpatches.Patch(color=colormap(i), label=f'Column {col}'))
                    
                    # シナプスの凡例
                    legend_handles.append(mpatches.Patch(color='red', label='Excitatory Synapses\n(Line width: Weight\nOpacity: Connection probability)'))
                    legend_handles.append(mpatches.Patch(color='blue', label='Inhibitory Synapses\n(Line width: Weight\nOpacity: Connection probability)'))
                    
                    plt.legend(handles=legend_handles, fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
                    
                    fig.canvas.draw_idle()
            
            # 状態のリセット
            dragged_node[0] = None
            mouse_moved[0] = False
            initial_click_pos[0] = None
        
        # マウスイベントの接続
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

    # タイトルと凡例
    plt.title("Network Structure", fontsize=24, pad=20, fontweight='bold', fontfamily='Times New Roman')
    
    # 凡例項目のリスト
    legend_handles = [mpatches.Patch(color='skyblue', label='Neurons')]
    
    # カラム情報がある場合は追加
    if column_numbers:
        import matplotlib.cm as cm
        colormap = cm.get_cmap('tab10', max(len(column_numbers), 1))
        for i, col in enumerate(sorted(column_numbers)):
            legend_handles.append(mpatches.Patch(color=colormap(i), label=f'Column {col}'))
    
    # シナプスの凡例
    legend_handles.append(mpatches.Patch(color='red', label='Excitatory Synapses\n(Line width: Weight\nOpacity: Connection probability)'))
    legend_handles.append(mpatches.Patch(color='blue', label='Inhibitory Synapses\n(Line width: Weight\nOpacity: Connection probability)'))
    
    plt.legend(handles=legend_handles, fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        mpld3.save_html(fig, save_path.replace(".png", ".html"))
    return fig
def assign_labels2neurons(spikemon, n_neuron:int, labels:list, input_labels:list, presentation_time, reset_time):
    """
    ニューロンにラベルを割り当てます。

    Args:
        spikemon (SpikeMonitor): スパイクモニター
        n_neuron (int): ニューロン数
        labels (list): 割り当てるラベルのリスト
        input_labels (list): 入力されたラベルのリスト
        presentation_time (float): 画像の提示時間
        reset_time (float): リセット時間

    Returns:
        np.ndarray: 各ニューロンに割り当てられたラベル
    """
    presentation_time /= ms
    reset_time /= ms
    interval_time = presentation_time + reset_time
    
    # スパイクデータをNumPy配列に変換
    spike_times = spikemon.t/ms if hasattr(spikemon.t, 'dim') else spikemon.t
    spike_indices = spikemon.i
    
    # 結果を格納する配列を初期化
    spike_cnt = np.zeros((n_neuron, len(labels)))
    
    # ラベルのインデックスを作成
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # 各画像の時間間隔を計算
    for n, label in enumerate(input_labels):
        start_time = n * interval_time
        end_time = (n + 1) * interval_time
        
        # 時間窓内のスパイクを一括で特定
        time_mask = (spike_times >= start_time) & (spike_times < end_time)
        active_neurons = spike_indices[time_mask]
        
        # 発火カウントを更新
        label_idx = label_to_idx[label]
        np.add.at(spike_cnt, (active_neurons, label_idx), 1)
    
    # 最も発火の多かったラベルのインデックスを取得し、対応するラベルを返す
    label_indices = np.argmax(spike_cnt, axis=1)
    return np.array([labels[idx] for idx in label_indices])

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

def get_population_rate(spikemon, start_time=0*ms, end_time=None):
    """
    集団発火率を計算します。

    Args:
        spikemon: SpikeMonitorオブジェクト
        start_time: 計算開始時間（デフォルト：0ms）
        end_time: 計算終了時間（デフォルト：モニタの最後の時間）

    Returns:
        発火率（Hz）
    """
    # 終了時間が指定されていない場合は最後のスパイク時間を使用
    if end_time is None:
        end_time = spikemon.t[-1]
    
    # ニューロン数の取得
    n_neurons = len(spikemon.count)
    
    # 時間範囲内のスパイクを取得（単位を揃える）
    spike_times_ms = spikemon.t/ms  # ミリ秒単位（単位なし）
    start_time_ms = start_time/ms   # ミリ秒単位（単位なし）
    end_time_ms = end_time/ms       # ミリ秒単位（単位なし）
    
    # 単位なしの値同士で比較
    spike_indices = np.where((spike_times_ms >= start_time_ms) & (spike_times_ms <= end_time_ms))[0]
    
    # スパイク数をカウント
    spike_count = len(spike_indices)
    
    # 時間範囲の計算（秒単位）
    duration = (end_time - start_time) / second
    
    # 0による除算を避ける
    if duration == 0 or n_neurons == 0:
        return 0 * Hz
    
    # population rate（Hz単位）の計算
    population_rate = spike_count / duration / n_neurons
    
    # Hzの単位を付けて返す
    return population_rate * Hz



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
def convert_value(value):
    """
    文字列値を単位付きの数値に変換します。

    Args:
        value: 変換する値
    Returns:
        変換後の値
    """
    if isinstance(value, str) and '*' in value:
        value_str, unit = value.split('*')
        return float(value_str) * eval(unit)
    return value

def process_parameters(data):
    """
    データ構造を再帰的に処理し、単位変換を行います。

    Args:
        data: 処理するデータ（辞書、リスト、または単一の値）
    Returns:
        変換後のデータ
    """
    if isinstance(data, dict):
        return {key: process_parameters(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [process_parameters(item) for item in data]
    else:
        return convert_value(data)

def load_parameters(file_path: str):
    """
    JSONファイルからパラメータを読み込みます。Brian2の単位変換も行います。

    Args:
        file_path (str): JSONファイルのパス
    Returns:
        parameters (dict): パラメータの辞書
    """
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    
    return process_parameters(loaded_data)
# def load_parameters(file_path:str):
#     """
#     JSONファイルからパラメータを読み込みます。Brian2の単位変換も行います。

#     Args:
#         file_path (str): JSONファイルのパス
#     Returns:
#         parameters (dict): パラメータの辞書
#     """
#     with open(file_path, "r") as f:
#         loaded_data = json.load(f)
#     for key_name in loaded_data.keys():
#         if isinstance(loaded_data[key_name], dict):
#             for key, value in loaded_data[key_name].items():
#                 if isinstance(value, str) and '*' in value:
#                     value, unit = value.split('*')
#                     loaded_data[key_name][key] = float(value) * eval(unit)
#         else:
#             if isinstance(loaded_data[key_name], str) and '*' in loaded_data[key_name]:
#                 value, unit = loaded_data[key_name].split('*')
#                 loaded_data[key_name] = float(value) * eval(unit)
#     return loaded_data

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
    pklファイルに保存されたモニターを読み込みます。以下のような方法で実際のモニター同様にデータを読み出せます。
    
    例：
    monitor = load_monitor(file_path)
    print(monitor.name)
    print(monitor.t)
    """
    try:
        with open(file_path, "rb") as f:
            try:
                monitor = pkl.load(f)
                # モニターデータの型チェック
                if not isinstance(monitor, (SpikeMonitorData, StateMonitorData)):
                    raise TypeError(f"ファイル {file_path} はモニターデータではありません。")
                return monitor
            except (pkl.UnpicklingError, EOFError) as e:
                raise ValueError(f"ファイル {file_path} の解析に失敗しました: {str(e)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイル {file_path} が見つかりません。")

# ===================================== 分割してモニターを保存 ==========================================
def save_all_monitors(network, save_path:str, index:int=None, compress=True):
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



