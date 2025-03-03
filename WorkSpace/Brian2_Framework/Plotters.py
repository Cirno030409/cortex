import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
import os
import glob
from matplotlib.animation import FuncAnimation
import Brian2_Framework.Tools as tools
import seaborn as sns
import matplotlib.animation as animation
import random
import mplcursors
from collections import deque
import time
    
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"

def population_rate_plot(populationratemonitors, time_end:int, time_start:int=0, save_path:str=None, smooth_window:int=None):
    """
    与えられたポピュレーションレートモニターから発火率のプロットを描画します。
    リストで複数のモニターを渡すと、それらを1枚のウィンドウにプロットします。
    表示する際には後ろにplt.show()が必要です。
    グラフを保存するには，保存するパスを渡します。
    発火率を平滑化する場合はsmooth_windowに平滑化するウィンドウサイズを渡します。
    シミュレーション全体での平均発火率も計算して表示します。

    Args:
        populationratemonitors (list): ポピュレーションレートモニターのリスト
        time_start (int): プロットする時間の開始範囲(ms) 
        time_end (int): プロットする時間の終了範囲(ms)
        save_path (str): 保存するパスの指定（オプション）
        smooth_window (int): 移動平均のウィンドウサイズ（オプション）。指定すると発火率を平滑化
    """
    if not isinstance(populationratemonitors, list):
        populationratemonitors = [populationratemonitors]
        
    # time_endとtime_startから単位を削除
    if hasattr(time_end, 'dimensions'):
        time_end = float(time_end/ms)
    if hasattr(time_start, 'dimensions'):
        time_start = float(time_start/ms)
        
    # 全体のレート計算のためのデータを準備
    all_times = []
    all_rates = []
    for mon in populationratemonitors:
        times = mon.t/ms
        rates = mon.rate/Hz
        mask = (times >= time_start) & (times <= time_end)
        all_times.append(times[mask])
        all_rates.append(rates[mask])
    
    # 共通の時間軸を作成（すべてのモニターの時間を結合してユニークな値を取得、ソート）
    if all_times:
        common_times = np.unique(np.concatenate(all_times))
        common_times.sort()
        
        # 全体のレート計算（時間ポイントごとの平均）
        total_rates = np.zeros_like(common_times)
        for i, t in enumerate(common_times):
            rates_at_t = []
            for mon_idx, times in enumerate(all_times):
                # 最も近い時間点を見つける
                idx = np.abs(times - t).argmin()
                if np.abs(times[idx] - t) < 1e-6:  # 十分に近い時間点があれば
                    rates_at_t.append(all_rates[mon_idx][idx])
            
            if rates_at_t:
                total_rates[i] = np.mean(rates_at_t)
        
    # ウィンドウの高さを計算（1つのプロットあたり2インチ）
    window_height = 2 * (len(populationratemonitors) + 1)  # +1 for total rate plot
    # ディスプレイの高さの80%を超える場合は、複数のウィンドウに分割
    max_height = plt.get_current_fig_manager().window.winfo_screenheight() * 0.8 / plt.rcParams['figure.dpi']
    n_windows = max(1, int(np.ceil(window_height / max_height)))
    
    # 全体のレートプロットは別のウィンドウに表示
    figs = []
    
    # 全体のレートプロット
    if all_times:
        fig_total = plt.figure(figsize=(14, 3))
        fig_total.canvas.manager.set_window_title("Total Population Rate")
        
        # スムージング処理
        if smooth_window is not None:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed_total_rates = np.convolve(total_rates, kernel, mode='same')
            line_smoothed = plt.plot(common_times, smoothed_total_rates, 'r-', label='Total (Smoothed)', linewidth=2)[0]
            line_original = plt.plot(common_times, total_rates, 'r-', alpha=0.3, label='Total (Original)')[0]
            plt.legend()
            
            # カーソルを追加（スムージング済みデータのみ）
            cursor_smoothed = mplcursors.cursor(line_smoothed, hover=True)
            
            @cursor_smoothed.connect("add")
            def on_add(sel):
                sel.annotation.set_text(f'Total (Smoothed)\nTime: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz')
        else:
            line = plt.plot(common_times, total_rates, 'r-', label='Total Rate', linewidth=2)[0]
            cursor = mplcursors.cursor(line, hover=True)
            
            @cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(f'Total\nTime: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz')
        
        # 全体の平均発火率を計算
        avg_rate = np.mean(total_rates)
        plt.axhline(y=avg_rate, color='k', linestyle='--', label=f'Avg: {avg_rate:.2f}Hz')
        plt.text(common_times[-1]*0.9, avg_rate*1.05, f'{avg_rate:.2f}Hz', fontsize=10)
        
        plt.ylabel("Rate (Hz)")
        plt.xlabel("Time (ms)")
        plt.title("Total Population Rate (All Neurons)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        figs.append(fig_total)
        
        # 全体レートを別ファイルとして保存
        if save_path:
            base, ext = os.path.splitext(save_path)
            total_save_path = f"{base}_total{ext}"
            fig_total.savefig(total_save_path)
    
    # 個々のポピュレーションのプロット
    monitors_per_window = int(np.ceil(len(populationratemonitors) / n_windows))
    for window_idx in range(n_windows):
        start_idx = window_idx * monitors_per_window
        end_idx = min((window_idx + 1) * monitors_per_window, len(populationratemonitors))
        current_monitors = populationratemonitors[start_idx:end_idx]
        
        fig = plt.figure(figsize=(14, 2*len(current_monitors)))
        
        # ウィンドウタイトルの生成
        if n_windows > 1:
            window_title = f"Population Rate Plot ({window_idx+1}/{n_windows})"
        else:
            window_title = "Population Rate Plot"
            
        fig.canvas.manager.set_window_title(window_title)
            
        for i, mon in enumerate(current_monitors):
            ax = plt.subplot(len(current_monitors), 1, i+1)
            
            # データの取得と時間範囲の制限
            times = mon.t/ms
            rates = mon.rate/Hz
            mask = (times >= time_start) & (times <= time_end)
            times = times[mask]
            rates = rates[mask]
            
            # スムージング処理
            if smooth_window is not None:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed_rates = np.convolve(rates, kernel, mode='same')
                line_smoothed = plt.plot(times, smoothed_rates, label='Smoothed')[0]
                line_original = plt.plot(times, rates, alpha=0.3, label='Original')[0]
                plt.legend()
                
                # カーソルを追加（スムージング済みデータのみ）
                cursor_smoothed = mplcursors.cursor(line_smoothed, hover=True)
                
                @cursor_smoothed.connect("add")
                def on_add(sel):
                    sel.annotation.set_text(f'Smoothed\nTime: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz')
            else:
                line = plt.plot(times, rates)[0]
                cursor = mplcursors.cursor(line, hover=True)
                
                @cursor.connect("add")
                def on_add(sel):
                    sel.annotation.set_text(f'Time: {sel.target[0]:.2f}ms\nRate: {sel.target[1]:.2f}Hz')
            
            # 平均発火率を計算して表示
            avg_rate = np.mean(rates)
            plt.axhline(y=avg_rate, color='k', linestyle='--', label=f'Avg: {avg_rate:.2f}Hz')
            plt.text(times[-1]*0.9, avg_rate*1.05, f'{avg_rate:.2f}Hz', fontsize=9)
            
            # モニター名からレイヤー情報を抽出
            if hasattr(mon, 'name'):
                name_parts = mon.name.split('_')
                layer_name = next((part for part in name_parts if part.startswith('L')), '')
                plt.ylabel(f"Rate (Hz)")
                plt.title(f"{mon.name}")  # モニター名を表示
            else:
                plt.ylabel("Rate (Hz)")
                plt.title(f"Monitor {i}")  # モニター番号を表示
                
            if i == len(current_monitors)-1:
                plt.xlabel("Time (ms)")
            
            plt.grid(True)
        
        plt.suptitle(window_title, fontsize=16)
        plt.tight_layout()
        figs.append(fig)
        
    if save_path:
        for i, fig in enumerate(figs[1:], 1):  # Skip the total rate plot which is already saved
            if n_windows > 1:
                base, ext = os.path.splitext(save_path)
                current_save_path = f"{base}_{i}{ext}"
            else:
                current_save_path = save_path
            fig.savefig(current_save_path)
            
    return figs

    
def raster_plot(spikemons, time_end:int, time_start:int=0, save_path:str=None):
    """
    与えられたスパイクモニターからラスタプロットを描画します。
    リストで複数のスパイクモニターを渡すと、それらを1枚のウィンドウにプロットします。
    表示する際には後ろにplt.show()が必要です。
    グラフを保存するには，保存するパスを渡します。

    Args:
        spikemons (list of SpikeMonitor): スパイクモニターのリスト
        time_start (int): プロットする時間の開始範囲(ms)
        time_end (int): プロットする時間の終了範囲(ms)
        save_path (str): 保存するパスの指定（オプション）
    """
    if not isinstance(spikemons, list):
        spikemons = [spikemons]
    # time_endとtime_startから単位を削除
    if hasattr(time_end, 'dimensions'):
        time_end = float(time_end/ms)
    if hasattr(time_start, 'dimensions'):
        time_start = float(time_start/ms)
        
    # ウィンドウの高さを計算（1つのプロットあたり2インチ）
    window_height = 2 * len(spikemons)
    # ディスプレイの高さの80%を超える場合は、複数のウィンドウに分割
    max_height = plt.get_current_fig_manager().window.winfo_screenheight() * 0.8 / plt.rcParams['figure.dpi']
    n_windows = max(1, int(np.ceil(window_height / max_height)))
    monitors_per_window = int(np.ceil(len(spikemons) / n_windows))
    
    figs = []
    for window_idx in range(n_windows):
        start_idx = window_idx * monitors_per_window
        end_idx = min((window_idx + 1) * monitors_per_window, len(spikemons))
        current_monitors = spikemons[start_idx:end_idx]
        
        fig = plt.figure(figsize=(14, 2*len(current_monitors)))
        
        # ウィンドウタイトルの生成
        if n_windows > 1:
            window_title = f"Raster plot ({window_idx+1}/{n_windows})"
        else:
            window_title = "Raster plot"
            
        # レイヤー情報の抽出（安全に処理）
        if all(hasattr(mon, 'name') for mon in current_monitors):
            layer_names = []
            for mon in current_monitors:
                # モニター名を_で分割
                name_parts = mon.name.split('_')
                # L1, L23, L4などの層名を探す
                layer = None
                for part in name_parts:
                    if part.startswith('L') and (part[1:].isdigit() or (len(part) > 2 and part[1:-1].isdigit())):
                        layer = part
                        break
                if layer is not None:
                    layer_names.append(layer)
            
            # 重複を除去してソート
            unique_layers = sorted(set(layer_names))
            if unique_layers:  # 層名が見つかった場合のみ追加
                window_title += f" - Layers: {', '.join(unique_layers)}"
        
        plt.suptitle(window_title)
        fig.canvas.manager.set_window_title(window_title)
        
        # サブプロットを作成
        axes = []
        scatter_plots = []
        for this_row in range(len(current_monitors)):
            if this_row == 0:
                ax = plt.subplot(len(current_monitors), 1, this_row+1)
            else:
                ax = plt.subplot(len(current_monitors), 1, this_row+1, sharex=axes[0])
            axes.append(ax)
            
            scatter = ax.scatter(current_monitors[this_row].t/ms, current_monitors[this_row].i, 
                               s=1, c='k', marker='.')
            scatter_plots.append(scatter)
            
            if this_row+1 == len(current_monitors):
                ax.set_xlabel('Time (ms)')
            ax.set_xlim(time_start, time_end)
            ax.set_ylim(-1, len(current_monitors[this_row].source))
            ax.set_ylabel('Neuron index')
            if hasattr(current_monitors[this_row], 'name'):
                ax.set_title(current_monitors[this_row].name)
            else:
                ax.set_title(f"Monitor {this_row}")
        
        # 各スキャッタープロットにカーソルを追加
        for scatter in scatter_plots:
            cursor = mplcursors.cursor(scatter, hover=False)
            
            @cursor.connect("add")
            def on_add(sel):
                neuron_idx = int(sel.target[1])
                time = sel.target[0]
                sel.annotation.set_text(f'Neuron: {neuron_idx}\nTime: {time:.2f}ms')
                sel.annotation.xy = (sel.target[0], sel.target[1])
        
        plt.subplots_adjust(hspace=0.7)
        plt.tight_layout()
        
        if save_path is not None:
            if n_windows > 1:
                base, ext = os.path.splitext(save_path)
                window_save_path = f"{base}_window{window_idx+1}{ext}"
            else:
                window_save_path = save_path
            plt.savefig(window_save_path)
        
        figs.append(fig)
    
    return figs

def state_plot(statemon:StateMonitor|list, time_end:int, time_start:int=0, neuron_num:int=0, variable_names:list=None, save_path:str=None):
    """
    与えられたステートモニターからプロットを描画します。この関数実行後にplt.show()などを記述する必要があります。
    変数のリストを渡すと，すべての変数のプロットを縦に並べて同に描画します。
    変数のリストを渡さない場合は，記録されているすべての変数をプロットします。
    IeとIi、geとgiがある場合は重ねて描画します。
    グラフを保存するには，保存するパスを渡します。
    
    Args:
        statemon (StateMonitor|list): ステートモニターまたはステートモニターのリスト
        neuron_num (int): プロットするニューロンの番号
        variable_names (list): プロットする変数の名前。Noneの場合は記録されているすべての変数をプロット
        fig_title (str): フィグのタイトル
        time_start (int): プロットする時間の開始範囲(ms)
        time_end (int): プロットする時間の終了範囲(ms)
    """        
    # time_endとtime_startから単位を削除
    if hasattr(time_end, 'dimensions'):
        time_end = float(time_end/ms)
    if hasattr(time_start, 'dimensions'):
        time_start = float(time_start/ms)
    
    # statemonがリストでない場合はリストに変換
    if not isinstance(statemon, list):
        statemon = [statemon]
    
    figs = []
    for monitor in statemon:
        # variable_namesがNoneの場合は記録されているすべての変数を使用
        if variable_names is None:
            monitor_vars = list(monitor.record_variables)
        else:
            monitor_vars = variable_names
        
        # IeとIi、geとgiのペアをチェック
        has_Ie = "Ie" in monitor_vars
        has_Ii = "Ii" in monitor_vars
        has_ge = "ge" in monitor_vars
        has_gi = "gi" in monitor_vars
        
        # プロットする変数リストを整理
        plot_vars = []
        if has_Ie:
            plot_vars.append(("Ie", ["Ie"]))
        if has_Ii:
            plot_vars.append(("Ii", ["Ii"]))
        if has_Ie or has_Ii:
            plot_vars.append(("I", ["Ie", "Ii"]))
            if has_Ie and has_Ii:  # 両方存在する場合は和も追加
                plot_vars.append(("I (E-I)", ["Ie-Ii"]))
        if has_ge or has_gi:
            plot_vars.append(("g", ["ge", "gi"]))
        for var in monitor_vars:
            if var not in ["Ie", "Ii", "ge", "gi"]:
                plot_vars.append((var, [var]))
                
        fig = plt.figure(figsize=(14, 3.0*len(plot_vars)))
        plt.suptitle(f"State plot - {monitor.name} - Neuron {neuron_num}")
        fig.canvas.manager.set_window_title(f"State plot - {monitor.name} - Neuron {neuron_num}")        
        # サブプロットを作成
        axes = []
        line_plots = []
        for this_row, (plot_name, vars_to_plot) in enumerate(plot_vars):
            if this_row == 0:
                ax = plt.subplot(len(plot_vars), 1, this_row+1)
            else:
                ax = plt.subplot(len(plot_vars), 1, this_row+1, sharex=axes[0])
            axes.append(ax)
            
            # 軸の範囲を固定するために、データの最小値と最大値を取得
            y_min = float('inf')
            y_max = float('-inf')
            has_valid_data = False

            # データの単位を取得・調整するための変数
            y_data_unit = None
            y_data_values = []

            for var in vars_to_plot:
                if var == "Ie-Ii" and "Ie" in monitor_vars and "Ii" in monitor_vars:
                    data = getattr(monitor, "Ie")[neuron_num] - getattr(monitor, "Ii")[neuron_num]
                    if len(data) > 0:
                        y_data_unit = data.get_best_unit()
                        data_scaled = data/y_data_unit
                        y_min = min(y_min, np.min(data_scaled))
                        y_max = max(y_max, np.max(data_scaled))
                        
                        line = ax.plot(monitor.t/ms, data_scaled, color="g", label="E-I")[0]
                        ax.axhline(y=0, color='gray', linestyle='--', alpha=1)
                        line_plots.append((line, var))
                        has_valid_data = True
                        y_data_values.extend(data_scaled)
                elif var in monitor_vars:
                    try:
                        data = getattr(monitor, var)[neuron_num]
                        if len(data) > 0:
                            if y_data_unit is None:
                                y_data_unit = data.get_best_unit()
                            data_scaled = data/y_data_unit
                            y_min = min(y_min, np.min(data_scaled))
                            y_max = max(y_max, np.max(data_scaled))
                            
                            if var in ["ge", "Ie"]:
                                color = "r"
                                label = "Excitatory"
                            elif var in ["gi", "Ii"]:
                                color = "b"
                                label = "Inhibitory"
                            else:
                                color = "k"
                                label = var
                            line = ax.plot(monitor.t/ms, data_scaled, color=color, label=label)[0]
                            line_plots.append((line, var))
                            has_valid_data = True
                            y_data_values.extend(data_scaled)
                    except (AttributeError, IndexError):
                        continue
            
            # データが存在する場合のみ軸の範囲とラベルを設定
            if has_valid_data:
                margin = (y_max - y_min) * 0.1
                if abs(y_max - y_min) < 1e-10:  # 値がほぼ同じ場合
                    y_min -= 0.1
                    y_max += 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
                
                # 単位を含む軸ラベルを設定
                unit_str = str(y_data_unit)
                if unit_str == '1':
                    ax.set_ylabel(plot_name)
                else:
                    ax.set_ylabel(f'{plot_name} ({unit_str})')
            else:
                # データが存在しない場合はデフォルトの範囲を設定
                ax.set_ylim(-1, 1)
                ax.text(0.5, 0.5, 'No data available', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)

            if len(vars_to_plot) > 1:
                ax.legend()
                
            if this_row+1 == len(plot_vars):
                ax.set_xlabel('Time (ms)')
            ax.set_xlim(time_start, time_end)
        
        # 各線プロットにカーソルを追加
        for line, var_name in line_plots:
            cursor = mplcursors.cursor(line, hover=False)
            
            @cursor.connect("add")
            def on_add(sel, var_name=var_name):
                time = sel.target[0]
                value = sel.target[1]
                sel.annotation.set_text(f'{var_name}\nTime: {time:.2f}ms\nValue: {value:.4f}')
                sel.annotation.xy = (sel.target[0], sel.target[1])
                
                # アノテーションの自動ズームを無効化
                ax = sel.artist.axes
                ax.autoscale(enable=False)
        
        plt.subplots_adjust(hspace=0.7)
        plt.tight_layout()
        if save_path is not None:
            # 複数のモニターがある場合はファイル名に番号を付加
            if len(statemon) > 1:
                base, ext = os.path.splitext(save_path)
                monitor_save_path = f"{base}_{monitor.name}{ext}"
            else:
                monitor_save_path = save_path
            plt.savefig(monitor_save_path)
        figs.append(fig)
    
    return figs
    return fig
    
def weight_plot_1_neuron(synapse, neuron_idx, n_pre, n_post):
    """
    与えられたシナプスグループから、指定されたニューロンの重みのマップを描画します。

    Args:
        synapse (SynapseGroup): シナプスグループ
        neuron_idx (int): ニューロンのインデックス
    """
    weight_mat = np.zeros((n_pre, n_post))
    for i, j, w in zip(synapse.i, synapse.j, synapse.w):
        weight_mat[i, j] = w
    
    weight_mat_plot = weight_mat[:, neuron_idx].reshape(int(np.sqrt(n_pre)), int(np.sqrt(n_pre)))
    plt.imshow(weight_mat_plot, cmap="viridis")

            
def weight_plot(synapse, n_pre, n_post, title="", save_fig=False, save_path:str=None, n_this_fig=0, assigned_labels=None):
    """
    与えられたステートモニターから重みのプロットを描画します。

    Args:
        synapse (SynapseGroup or list or np.ndarray): シナプスグループか重みの行列
        n_pre (int): 前のニューロンの数
        n_post (int): 後のニューロンの数
        save_fig (bool): フィグを保存するかどうか
        save_path (str): フィグを保存するパス
        n_this_fig (int): このフィグの番号（保存するのファイル名にな)
        assigned_labels (list): 割り当てられたラベルのリスト
    """
    # synapse.w[neuron_idx][time_idx]
    weight_mat = np.zeros((n_pre, n_post))
    if isinstance(synapse, Synapses):
        for i, j, w in zip(synapse.i, synapse.j, synapse.w):
            weight_mat[i, j] = w
    elif isinstance(synapse, list) or isinstance(synapse, np.ndarray):
        weight_mat = np.array(synapse)
        weight_mat = weight_mat.reshape(n_pre, n_post)
    else:
        raise ValueError("synapseはSynapsesクラスのインスタンスか，重みの行列である必要があります。")
        
    # サブプロットの行数と列数を計算
    n_rows = int(np.ceil(np.sqrt(n_post)))
    n_cols = int(np.ceil(n_post / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 12))  # 幅を10に増やして右側に空白を作る
    axes = axes.flatten()  # 2次元配列を1次元に変換

    for img in range(n_post):
        weightloop = weight_mat[:, img].reshape(
            int(np.sqrt(n_pre)), int(np.sqrt(n_pre))
        )
        cax = axes[img].matshow(weightloop, cmap="viridis")
        axes[img].set_xticks([])
        axes[img].set_yticks([])
        
        if assigned_labels is not None:
            axes[img].set_xlabel(f"Neuron {img},label:{assigned_labels[img]}", fontsize=6)
        else:
            axes[img].set_xlabel(f"Neuron {img}", fontsize=6)

    # 余分なサブプロットを非表にする
    for img in range(n_post, n_rows * n_cols):
        axes[img].axis('off')

    fig.suptitle(f"Weight plot - {synapse.name}")
    fig.canvas.manager.set_window_title(f"Weight plot - {synapse.name}")

    # 右側にカラーバーを配置
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 右側に空白を作り、そこにカラーバーを配置
    plt.colorbar(cax, cax=cbar_ax)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) # カラーバーの警告を無視
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 右側に10%の空白を確保
    warnings.filterwarnings("default", category=UserWarning) # カラーバーの警告を有効化
    
    if save_path is not None:
        if n_this_fig == 0:
            # save_path内の全ファイルを削除
            files = glob.glob(os.path.join(save_path, '*.png'))
            for f in files:
                os.remove(f)
                print(f"\tDeleted {f}")
        plt.savefig(save_path + f"{n_this_fig}.png")
        
        plt.clf()
        plt.close()   
        
def firing_rate_heatmap(spikemon, start_time, end_time, save_fig=False, save_path:str=None, n_this_fig=None):
    """
    与えられたスパイクモニターから発火率のヒートマップを描画します。

    Args:
        spikemon (SpikeMonitor): スパイクモニター
        start_time (float): 開始時間
        end_time (float): 終了時間
        save_fig (bool): フィグを保存するかどうか
        save_path (str): フィグを保存するパス
        n_this_fig (int): のフィグの番号（保存する際のファイル名になる)
    Returns:
        heatmap_data (np.ndarray): 発火率のヒートマップのデータ
    """

    firing_rates = tools.get_firing_rate(spikemon, start_time, end_time, mode="count")
    n_neurons = len(firing_rates)
    
    # 最適な行数と列数を計算
    n_rows = int(np.ceil(np.sqrt(n_neurons)))
    n_cols = int(np.ceil(n_neurons / n_rows))
    
    heatmap_data = np.full((n_rows, n_cols), np.nan)
    for i in range(n_neurons):
        row = i // n_cols
        col = i % n_cols
        heatmap_data[row, col] = firing_rates[i]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt='.2f', cbar=True, mask=np.isnan(heatmap_data), vmin=0)
    plt.title(f'Firing Count Heatmap ({start_time} to {end_time})')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{n_this_fig}_count_heatmap.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.clf()
    plt.close()
    
    return heatmap_data

def visualize_wta_response(input_image, synapse, spikemon, start_time:int, exposure_time:int, save_path=None, n_this_fig=None):
    """
    Parameters:
    -----------
    input_image : np.array (height, width)
        入力画像
    synapse : SynapseGroup
        シナプスグループ
    spikemon : SpikeMonitor
        スパイクモニター
    start_time : int
        開始時間
    exposure_time : int
        露光時間
    """
    n_i = synapse.N_incoming_post[0]
    n_j = synapse.N_outgoing_pre[0]
    
    weight_mat = np.zeros((n_i, n_j))
    for i, j, w in zip(synapse.i, synapse.j, synapse.w):
        weight_mat[i, j] = w
    # VariableViewをNumPy配列に変換
    weights = np.array(weight_mat)

    # スパイクカウントの計算
    n_neurons = n_j
    spike_counts = np.zeros(n_neurons)
    # start_timeからexposure_time経過後までのスパイクのみを集計
    end_time = start_time + exposure_time
    spikes = tools.get_spikes_within_time_range(spikemon=spikemon, start_time=start_time, end_time=end_time)
    spike_indices = [spike[1] for spike in spikes]
    unique, counts = np.unique(spike_indices, return_counts=True)
    spike_counts[unique] = counts
    
    # 上位5位までのニューロンを特定
    top_5_winners = np.argsort(spike_counts)[::-1][:5] if len(spike_counts) > 0 else [0] * 5
    
    # プロットのサイズを調整
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(5, 5, figure=fig)
    
    # 1. 入力画像 (左上)
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(input_image, cmap='gray')
    ax_input.set_title('Input Image')
    ax_input.axis('off')
    
    # 2. 上位5位のニューロンの重み (右側)
    for i, winner in enumerate(top_5_winners):
        ax = fig.add_subplot(gs[i, 1])
        winner_weights = weights[:, winner].reshape((int(np.sqrt(n_i)), int(np.sqrt(n_i))))
        ax.imshow(winner_weights, cmap='gray')
        ax.set_title(f'#{i+1} Neuron(#{winner})\nSpike Count: {spike_counts[winner]:.0f}')
        ax.axis('off')
    
    # 3. 1位のニューロンの重みと入力の重なり (左中央)
    for i, winner in enumerate(top_5_winners): 
        ax = fig.add_subplot(gs[i, 2])
        winner_weights = weights[:, winner].reshape((int(np.sqrt(n_i)), int(np.sqrt(n_i))))
        overlap = input_image * winner_weights
        ax.imshow(overlap, cmap='coolwarm')
        ax.set_title(f'#{i+1} Neuron(#{winner})\nOverlap of Input and Weight\n(Red: Positive, Blue: Negative)')
        ax.axis('off')
    
    # 4. スパイク発火頻度 (左下)
    ax_spikes = fig.add_subplot(gs[1:4, 3:5])
    bar_colors = ['gray'] * n_neurons  # デフォルトは全てグレー

    # 上位5位のニューロンの色を設定
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    for winner, color in zip(top_5_winners, colors):
        bar_colors[winner] = color

    # 色分けされた棒グラフを描画
    bars = ax_spikes.bar(range(n_neurons), spike_counts, color=bar_colors)
    ax_spikes.set_title('Spike Count per Neuron')
    ax_spikes.set_xlabel('Neuron ID')
    ax_spikes.set_xticks(np.arange(0, n_neurons+1, 10))
    ax_spikes.set_ylabel('Spike Count')

    # 凡例を追加
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=f'#{i+1} Winner (Neuron #{winner})')
                        for i, (winner, color) in enumerate(zip(top_5_winners, colors))]
    ax_spikes.legend(handles=legend_elements)

    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'wta_response_{n_this_fig}.png'))
        plt.close()
    
    return fig

def plot_all_monitors(network, time_end:int, time_start:int=0, save_path:str=None, smooth_window:int=None):
    """
    ネットワーク内のすべてのスパイクモニター、ステートモニター、ポピュレーションレートモニターを描画します。

    Args:
        network (dict or Network): ネットワークの辞書またはNetworkオブジェクト
        time_end (int): プロットする時間の終了範囲(ms)
        time_start (int): プロットする時間の開始範囲(ms)
        save_path (str): 保存するパスの指定（オプション）
    """
    # 各種モニターを分類
    spike_monitors = []
    state_monitors = []
    population_rate_monitors = []
    
    # ネットワークがdictの場合とNetworkオブジェクトの場合で処理を分ける
    if hasattr(network, 'items'):  # dictの場合
        items = network.items()
    else:  # Networkオブジェクトの場合
        items = [(obj.name, obj) for obj in network.objects if hasattr(obj, 'name')]
    
    for key, value in items:
        if isinstance(key, str):  # キーが文字列の場合
            if "spikemon" in key:
                spike_monitors.append(value)
            elif "statemon" in key:
                state_monitors.append(value)
            elif "popmon" in key:
                population_rate_monitors.append(value)
        else:  # キーが文字列でない場合は、オブジェクトの型で判断
            from brian2.monitors import SpikeMonitor, StateMonitor, PopulationRateMonitor
            if isinstance(value, SpikeMonitor):
                spike_monitors.append(value)
            elif isinstance(value, StateMonitor):
                state_monitors.append(value)
            elif isinstance(value, PopulationRateMonitor):
                population_rate_monitors.append(value)
    
    # スパイクモニターを層ごとにグループ化
    layer_groups = {"other": []}  # 層名がないモニターはotherグループに入れる
    
    for monitor in spike_monitors:
        if hasattr(monitor, 'name'):
            # モニター名を_で分割
            name_parts = monitor.name.split('_')
            # L1, L23, L4などの層名を探す
            layer = None
            for part in name_parts:
                if part.startswith('L') and (part[1:].isdigit() or (len(part) > 2 and part[1:-1].isdigit())):
                    layer = part
                    break
            
            if layer is not None:
                if layer not in layer_groups:
                    layer_groups[layer] = []
                layer_groups[layer].append(monitor)
            else:
                layer_groups["other"].append(monitor)
        else:
            layer_groups["other"].append(monitor)
    
    # 空のグループを削除
    layer_groups = {k: v for k, v in layer_groups.items() if v}
    
    # 層ごとにソートしてプロット（otherは最後に）
    figs = []
    # まず層を持つグループを処理
    sorted_layers = sorted([layer for layer in layer_groups.keys() if layer != "other"])
    
    # 層を持つグループを処理
    for layer in sorted_layers:
        if save_path is not None:
            layer_save_path = os.path.join(save_path, f'raster_plot_{layer}.png')
        else:
            layer_save_path = None
        figs.extend(raster_plot(layer_groups[layer], time_end=time_end, time_start=time_start, save_path=layer_save_path))
    
    # otherグループを処理
    if "other" in layer_groups:
        if save_path is not None:
            other_save_path = os.path.join(save_path, 'raster_plot_other.png')
        else:
            other_save_path = None
        figs.extend(raster_plot(layer_groups["other"], time_end=time_end, time_start=time_start, save_path=other_save_path))
    
    # ステートモニターをプロット
    if state_monitors:
        if save_path is not None:
            state_save_path = os.path.join(save_path, 'state_plot.png')
        else:
            state_save_path = None
        figs.extend(state_plot(state_monitors, time_end=time_end, time_start=time_start, save_path=state_save_path))

    # ポピュレーションレートモニターをプロット
    if population_rate_monitors:
        if save_path is not None:
            poprate_save_path = os.path.join(save_path, 'population_rate_plot.png')
        else:
            poprate_save_path = None
        figs.extend(population_rate_plot(population_rate_monitors, time_end=time_end, time_start=time_start, save_path=poprate_save_path, smooth_window=smooth_window))
    
    return figs