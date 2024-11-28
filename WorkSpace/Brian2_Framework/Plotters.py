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
import threading
import queue
from collections import deque
import time

class Common_Plotter:
    """
    シミュレーション後のネットワークの記録したデータのプロットを行う。
    """        
    
    def __init__(self):
        """
        様々なグラフを描画するプロッターを作成します。
        シミュレーション時間は Brian2.Network.t を渡す必要があります。
        
        Methods:
            raster_plot(spikemon, all_rows, this_row, fig_title=""): 
                スパイクモニターからラスタプロットを描画します。
            
            state_plot(statemon, neuron_num, variable_name, all_rows, this_row, fig_title=""): 
                ステートモニターからプロットを描画します。
            
            weight_plot(synapse, n_pre, n_post, title="", save_fig=False, save_path="", n_this_fig=0): 
                シナプスグループから重みのプロットを描画します。

        """
        self.simu_time = None
        
    def set_simu_time(self, simu_time):
        """
        シミュレーションの時間を設定します。
        
        Args:
            simu_time (float): シミュレーションの時間
        """
        self.simu_time = simu_time
        
    def raster_plot(self, spikemons:list, time_start:int=0, time_end:int=None, save_path:str=None):
        """
        与えられたスパイクモニターからラスタプロットを描画します。
        リストで複数のスパイクモニターを渡すと、それらを1枚のウィンドウにプロットします。
        表示する際には後ろにplt.show()が必要です。
        グラフを保存するには，保存するパスを渡します。

        Args:
            spikemons (list of SpikeMonitor): スパイクモニターのリスト
            fig_title (str): フィグのタイトル
            time_start (int): プロットする時間の開始範囲(ms)
            time_end (int): プロットする時間の終了範囲(ms)
        """
        if time_end is None:
            time_end = self.simu_time*1000
        fig = plt.figure(figsize=(14, 2*len(spikemons)))
        plt.suptitle(f"Raster plot")
        fig.canvas.manager.set_window_title(f"Raster plot - {spikemons[0].name}")
        all_rows = len(spikemons)
        if self.simu_time is None and time_end is None:
            raise ValueError("time_end引数を用いてプロット時間を限定するか，予めset_simu_time()を使用してシミュレーション時間を設定してください。")
            
        # サブプロットを作成
        axes = []
        scatter_plots = []  # スキャッタープロットを保存するリスト
        for this_row in range(all_rows):
            if this_row == 0:
                ax = plt.subplot(all_rows, 1, this_row+1)
            else:
                ax = plt.subplot(all_rows, 1, this_row+1, sharex=axes[0])
            axes.append(ax)
            
            scatter = ax.scatter(spikemons[this_row].t/ms, spikemons[this_row].i, 
                               s=1, c='k', marker='.')
            scatter_plots.append(scatter)
            
            if this_row+1 == all_rows:
                ax.set_xlabel('Time (ms)')
            ax.set_xlim(time_start, time_end)
            ax.set_ylim(0, len(spikemons[this_row].source))
            ax.set_ylabel('Neuron index')
            ax.set_title(spikemons[this_row].name)
        
        # 各スキャッタープロットにカーソルを追加
        for i, scatter in enumerate(scatter_plots):
            cursor = mplcursors.cursor(scatter, hover=False)
            
            @cursor.connect("add")
            def on_add(sel):
                # アノテーションのテキストをカスタマイズ
                neuron_idx = int(sel.target[1])  # Y座標（ニューロン番号）
                time = sel.target[0]  # X座標（時間）
                sel.annotation.set_text(f'Neuron: {neuron_idx}\nTime: {time:.2f}ms')
                # アノテーションの位置を調整（必要に応じて）
                sel.annotation.xy = (sel.target[0], sel.target[1])
        
        plt.subplots_adjust(hspace=0.7)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    def state_plot(self, statemon:StateMonitor, neuron_num:int, variable_names:list, time_start:int=0, time_end:int=None, save_path:str=None):
        """
        与えられたステートモニターからプロットを描画します。この関数実行後にplt.show()などを記述する必要があります。
        変数のリストを渡すと，すべての変数のプロットを縦に並べて同に描画します。
        IeとIi、geとgiがある場合は重ねて描画します。
        グラフを保存するには，保存するパスを渡します。
        
        Args:
            statemon (StateMonitor): ステートモニター
            neuron_num (int): プロットするニューロンの番号
            variable_names (list): プロットする変数の名前
            fig_title (str): フィグのタイトル
            time_start (int): プロットする時間の開始範囲(ms)
            time_end (int): プロットする時間の終了範囲(ms)
        """
        if time_end is None:
            time_end = self.simu_time*1000
            
        # IeとIi、geとgiのペアをチェック
        has_Ie = "Ie" in variable_names
        has_Ii = "Ii" in variable_names
        has_ge = "ge" in variable_names
        has_gi = "gi" in variable_names
        
        # プロットする変数リストを整理
        plot_vars = []
        if has_Ie:
            plot_vars.append(("Ie", ["Ie"]))
        if has_Ii:
            plot_vars.append(("Ii", ["Ii"]))
        if has_Ie or has_Ii:
            plot_vars.append(("I", ["Ie", "Ii"]))
            if has_Ie and has_Ii:  # 両方存在する場合は和も追加
                plot_vars.append(("I (E+I)", ["Ie+Ii"]))
        if has_ge or has_gi:
            plot_vars.append(("g", ["ge", "gi"]))
        for var in variable_names:
            if var not in ["Ie", "Ii", "ge", "gi"]:
                plot_vars.append((var, [var]))
                
        fig = plt.figure(figsize=(14, 1.3*len(plot_vars)))
        plt.suptitle(f"State plot - {statemon.name} - Neuron {neuron_num}")
        fig.canvas.manager.set_window_title(f"State plot - {statemon.name} - Neuron {neuron_num}")
        
        if self.simu_time is None and time_end is None:
            raise ValueError("time_end引数を用いてプロット時間を限定するか，予めset_simu_time()を使用してシミュレーション時間を設定してください。")
            
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

            for var in vars_to_plot:
                if var == "Ie+Ii" and "Ie" in variable_names and "Ii" in variable_names:
                    # Ie + Iiの和を計算
                    data = getattr(statemon, "Ie")[neuron_num] + getattr(statemon, "Ii")[neuron_num]
                    if len(data) > 0:  # データが存在する場合のみ処理
                        y_min = min(y_min, np.min(data))
                        y_max = max(y_max, np.max(data))
                        line = ax.plot(statemon.t/ms, data, color="g", label="E+I")[0]
                        ax.axhline(y=0, color='gray', linestyle='--', alpha=1)
                        line_plots.append((line, var))
                        has_valid_data = True
                elif var in variable_names:
                    try:
                        data = getattr(statemon, var)[neuron_num]
                        if len(data) > 0:  # データが存在する場合のみ処理
                            y_min = min(y_min, np.min(data))
                            y_max = max(y_max, np.max(data))
                            
                            if var in ["ge", "Ie"]:
                                color = "r"
                                label = "Excitatory"
                            elif var in ["gi", "Ii"]:
                                color = "b"
                                label = "Inhibitory"
                            else:
                                color = "k"
                                label = var
                            line = ax.plot(statemon.t/ms, data, color=color, label=label)[0]
                            line_plots.append((line, var))
                            has_valid_data = True
                    except (AttributeError, IndexError):
                        continue
            
            # データが存在する場合のみ軸の範囲を設定
            if has_valid_data:
                margin = (y_max - y_min) * 0.1
                if abs(y_max - y_min) < 1e-10:  # 値がほぼ同じ場合
                    y_min -= 0.1
                    y_max += 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
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
            ax.set_ylabel(plot_name)
        
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
            plt.savefig(save_path)
        return fig
        
    def weight_plot_1_neuron(self, synapse, neuron_idx, n_pre, n_post):
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

                
    def weight_plot(self, synapse, n_pre, n_post, title="", save_fig=False, save_path:str=None, n_this_fig=0, assigned_labels=None):
        """
        与えられたステートモニターから重みのプロットを描画します。

        Args:
            synapse (SynapseGroup): シナプスグループ
            n_pre (int): 前のニューロンの数
            n_post (int): 後のニューロンの数
            save_fig (bool): フィグを保存するかどうか
            save_path (str): フィグを保存するパス
            n_this_fig (int): このフィグの番号（保存するのファイル名にな)
            assigned_labels (list): 割り当てられたラベルのリスト
        """
        # synapse.w[neuron_idx][time_idx]
        weight_mat = np.zeros((n_pre, n_post))
        for i, j, w in zip(synapse.i, synapse.j, synapse.w):
            weight_mat[i, j] = w
            
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
            
    def firing_rate_heatmap(self, spikemon, start_time, end_time, save_fig=False, save_path:str=None, n_this_fig=None):
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
    
    def visualize_wta_response(self, input_image, synapse, spikemon, start_time:int, exposure_time:int, save_path=None, n_this_fig=None):
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
        