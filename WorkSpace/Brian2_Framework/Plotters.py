import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
import os
import glob
import Brian2_Framework.Tools as tools
import seaborn as sns
import matplotlib.animation as animation
import random
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
        plt.figure(figsize=(15, 2*len(spikemons)))
        plt.suptitle("Raster plot")
        all_rows = len(spikemons)
        if self.simu_time is None:
            raise ValueError("シミュレーション時間が設定されていません。set_simu_time()を使用してシミュレーション時間を設定してください。")
        for this_row in range(all_rows):
            plt.subplot(all_rows, 1, this_row+1)
            plt.plot(spikemons[this_row].t/ms, spikemons[this_row].i, '.k', markersize=1)
            if this_row+1 == all_rows:
                plt.xlabel('Time (ms)')
            plt.xlim(time_start, time_end)
            plt.ylim(0, len(spikemons[this_row].source))
            plt.ylabel('Neuron index')
            plt.title(spikemons[this_row].name)
        plt.subplots_adjust(hspace=0.7)
        if save_path is not None:
            plt.savefig(save_path)

    def state_plot(self, statemon:StateMonitor, neuron_num:int, variable_names:list, time_start:int=0, time_end:int=None, save_path:str=None):
        """
        与えられたステートモニターからプロットを描画します。この関数実行後にplt.show()などを記述する必要があります。
        変数のリストを渡すと，すべての変数のプロットを縦に並べて同時に描画します。
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
        plt.figure(figsize=(15, 2*len(variable_names)))
        plt.suptitle(f"State plot - {statemon.name}")
        all_rows = len(variable_names)
        if self.simu_time is None:
            raise ValueError("シミュレーション時間が設定されていません。set_simu_time()を使用して全体のシミュレーション時間を設定してください。")
        for this_row in range(all_rows):
            plt.subplot(all_rows, 1, this_row+1)
            if variable_names[this_row] == "ge":
                color = "r"
            elif variable_names[this_row] == "gi":
                color = "b"
            else:
                color = "k"
            plt.plot(statemon.t/ms, getattr(statemon, variable_names[this_row])[neuron_num], color=color)
            if this_row+1 == all_rows:
                plt.xlabel('Time (ms)')
            plt.xlim(time_start, time_end)
            plt.ylabel(variable_names[this_row])
        plt.subplots_adjust(hspace=0.7)
        if save_path is not None:
            plt.savefig(save_path)
        
    def raster_plot_time_window(self, spikemon:SpikeMonitor, all_rows:int, this_row:int, time_window_size:int, fig_title:str=""):
        # TODO 実装途中
        """
        与えられたスパイクモニターからリアルタイムでラスタプロットを描画します。
        使用するには、メインループ内にplt.show(block=False)とplt.pause(0.1)の記述が必要です。

        Args:
            spikemon (SpikeMonitor): スパイクモニター
            all_rows (int): 縦に並べるラスタープロットの数
            this_row (int): このプロットを設置する行
            time_window_size (int): 時間窓のサイズ(ms)
        """
        fig, ax = plt.subplots(all_rows, 1)
        xlim = [0, time_window_size]
        X, Y = [], []
        def update(frame):
            # global X, Y
            print(len(spikemon.i))
            Y.append(random.random())
            X.append(len(Y))
            # if len(spikemon.i) > 0: 
            #     # Y.append(spikemon.i[-1])?
            #     X.append(len(Y))
            # else:
            #     X.append(0)
            #     Y.append(0)
            
            if len(X) > time_window_size:
                xlim[0] += 1
                xlim[1] += 1
                
            ax.clear()
            line, = ax.plot(X, Y)
            ax.set_title(fig_title)
            ax.set_ylim(0, len(spikemon.source))
            ax.set_xlim(xlim[0], xlim[1])
            
            return [line]
            
        ani = animation.FuncAnimation(fig, update, interval=10, blit=True)
        
        
        
        
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
            n_this_fig (int): このフィグの番号（保存する際のファイル名になる)
            assigned_labels (list): 割り当てられたラベルのリスト
        """
        # synapse.w[neuron_idx][time_idx]
        weight_mat = np.zeros((n_pre, n_post))
        for i, j, w in zip(synapse.i, synapse.j, synapse.w):
            weight_mat[i, j] = w
            
        # サブプロットの行数と列数を計算
        n_rows = int(np.ceil(np.sqrt(n_post)))
        n_cols = int(np.ceil(n_post / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 9))  # 幅を10に増やして右側に空白を作る
        axes = axes.flatten()  # 2次元配列を1次元に変換

        for img in range(n_post):
            weightloop = weight_mat[:, img].reshape(
                int(np.sqrt(n_pre)), int(np.sqrt(n_pre))
            )
            cax = axes[img].matshow(weightloop, cmap="viridis")
            axes[img].set_xticks([])
            axes[img].set_yticks([])
            
            if assigned_labels is not None:
                axes[img].set_xlabel(f"{assigned_labels[img]}", fontsize=8)

        # 余分なサブプロットを非表示にする
        for img in range(n_post, n_rows * n_cols):
            axes[img].axis('off')

        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

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
            n_this_fig (int): このフィグの番号（保存する際のファイル名になる)
        """

        firing_rates = tools.get_firing_rate(spikemon, start_time, end_time)
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
        plt.title(f'Firing Rate Heatmap ({start_time} to {end_time})')
        plt.xlabel('Neuron Index')
        plt.ylabel('Neuron Index')
        
        if save_path is not None:
            plt.savefig(f'{save_path}{n_this_fig}_rate_heatmap.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.clf()
        plt.close()
class Plotter:
    """
    プロットを行うクラス．
    """
    def __init__(self, network, column_id:int):
        self.network = network.obj
        self.v_th_for_plot_l4 = network["N_l4"].namespace["v_th"]
        self.v_rest_for_plot_l4 = network["N_l4"].namespace["v_reset"]
        
        self.v_th_for_plot_l23 = network["N_l23"].namespace["v_th"]
        self.v_rest_for_plot_l23 = network["N_l23"].namespace["v_reset"]
        
        self.n_l4 = network.n_l4
        self.n_l23 = network.n_l23
        self.n_inp = network.n_inp
        
        self.S = {}
        self.S["l4->l23"] = network["S_l4_l23"]
        
        self.column_id = column_id
        
        self.statemon = {}
        self.spikemon = {}
        
        self.statemon["l4"] = network["statemon_N_l4"]
        self.statemon["l23"] = network["statemon_N_l23"]
        self.statemon["S_l4->l23"] = network["statemon_S_l4_l23"]
        self.spikemon["l23"] = network["spikemon_N_l23"]
        self.spikemon["l4"] = network["spikemon_N_l4"]
        self.spikemon["input"] = network["spikemon_N_input"]

    def draw_potential(
        self, neuron_num_l4=None, neuron_num_l23=None, title: str = "Membrane potential"
    ):
        """
        カラム内のすべてのニューロンの膜電位をプロットする．この関数呼び出し後にplt.show()の記述が必要．

        Args:
            neuron_num_l4 (list of int): プロットするニューロンの番号のリスト. Defaults to None.
            neuron_num_l23 (list of int): プロットするニューロンの番号のリスト. Defaults to None.
            title (str, optional): グラフタイトル. Defaults to "All membrane potential".
        """
        if neuron_num_l4 is None:
            neurons = range(self.n_l4)
        else:
            neurons = neuron_num_l4
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id}) [L4]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i, j in enumerate(neurons):
            ax[i].plot(self.statemon["l4"].t / ms, self.statemon["l4"].v[j], color="k")
            ax[i].set_ylabel(f"Neuron No.{j}")
            ax[i].set_ylim(self.v_rest_for_plot_l4 - 20, self.v_th_for_plot_l4 + 20)
            ax[i].axhline(
                self.v_rest_for_plot_l4,
                color="red",
                linewidth=0.5,
                linestyle="--",
                label="Resting Potential",
            )
            ax[i].axhline(
                self.v_th_for_plot_l4,
                color="blue",
                linewidth=0.5,
                linestyle="--",
                label="Threshold Potential",
            )
            ax[i].legend(loc="upper left")
        fig.suptitle(title + subtitle)

        if neuron_num_l23 is None:
            neurons = range(self.n_l23)
        else:
            neurons = neuron_num_l23
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id}) [L2/3]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i, j in enumerate(neurons):
            ax[i].plot(
                self.statemon["l23"].t / ms, self.statemon["l23"].v[j], color="k"
            )
            ax[i].set_ylabel(f"Neuron No.{j}")
            ax[i].set_ylim(self.v_rest_for_plot_l23 - 20, self.v_th_for_plot_l23 + 20)
            ax[i].axhline(
                self.v_rest_for_plot_l23,
                color="red",
                linewidth=0.5,
                linestyle="--",
                label="Resting Potential",
            )
            ax[i].axhline(
                self.v_th_for_plot_l23,
                color="blue",
                linewidth=0.5,
                linestyle="--",
                label="Threshold Potential",
            )
            ax[i].legend()
        fig.suptitle(title + subtitle)

    def draw_current(
        self, neuron_num_l4=None, neuron_num_l23=None, title: str = "Current"
    ):
        """
        カラム内のすべてのニューロンの電流をプロットする．この関数呼び出し後にplt.show()の記述が必要．

        Args:
            neuron_num_l4 (list of int): プロットするニューロンの番号のリスト. 記述しないとすべてのニューロンをプロットする.
            neuron_num_l23 (list of int): プロットするニューロンの番号のリスト. 記述しないとすべてのニューロンをプロットする.
            title (str, optional): グラフタイトル. Defaults to "All current".
        """
        if neuron_num_l4 is None:
            neurons = range(self.n_l4)
        else:
            neurons = neuron_num_l4
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id}) [L4]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i, j in enumerate(neurons):
            ax[i].plot(self.network["statemon_N_l4"].t / ms, self.network["statemon_N_l4"].Ie[j], color="r", label="Excitatory")
            ax[i].plot(self.network["statemon_N_l4"].t / ms, self.network["statemon_N_l4"].Ii[j], color="b", label="Inhibitory")
            ax[i].set_ylabel(f"Neuron No.{j}")
            ax[i].set_ylim(0, 120)
            ax[i].set_xlabel("Time (ms)")
            ax[i].legend()
        fig.suptitle(title)

        if neuron_num_l23 is None:
            neurons = range(self.n_l23)
        else:
            neurons = neuron_num_l23
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id}) [L2/3]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i, j in enumerate(neurons):
            ax1 = ax[i]
            ax2 = ax1.twinx()
            ax1.plot(
                self.network["statemon_N_l23"].t / ms, self.network["statemon_N_l23"].Ie[j], color="r", label="Excitatory"
            )
            ax2.plot(
                self.network["statemon_N_l23"].t / ms, self.network["statemon_N_l23"].Ii[j], color="b", label="Inhibitory"
            )
            ax1.set_ylabel(f"Neuron No.{j}")
            ax1.set_ylim(-10, 120)
            ax2.set_ylim(10, -120)
            ax1.set_xlabel("Time (ms)")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
        fig.suptitle(title)
        
        try:
            N_inh = self.network["N_inh"]
        except KeyError:
            N_inh = None
        if N_inh is not None:
            fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
            subtitle = f" (column_id: {self.column_id}) [Inhibitory]"
            fig.canvas.manager.set_window_title(title + subtitle)
            for i, j in enumerate(neurons):
                ax1 = ax[i]
                ax2 = ax1.twinx()
                ax1.plot(
                    self.network["statemon_N_inh"].t / ms, self.network["statemon_N_inh"].Ie[j], color="r", label="Excitatory"
                )
                ax2.plot(
                    self.network["statemon_N_inh"].t / ms, self.network["statemon_N_inh"].Ii[j], color="b", label="Inhibitory"
                )
                ax1.set_ylabel(f"Neuron No.{j}")
                ax1.set_ylim(-10, 120)
                ax2.set_ylim(10, -120)
                ax1.set_xlabel("Time (ms)")
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")
            fig.suptitle(title + subtitle)
            
    def draw_threshold_changes(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
        title = "Threshold[L4]"
        subtitle = f" (column_id: {self.column_id}) [Threshold]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i in range(1):
            ax[i].plot(self.network["statemon_N_l4"].t / ms, self.network["statemon_N_l4"].theta[15], color="k")
            ax[i].set_ylabel(f"Neuron No.{i}")
            ax[i].set_xlabel("Time (ms)")
        fig.suptitle(title + subtitle)

    def draw_conductance(
        self, title: str = "Conductance"
    ):
        """
        シナプスモデルのコンダクタンスのグラフを描画する．
        """
        fig, ax = plt.subplots(self.network.n_l23 + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id}) [L4->L2/3]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i in range(self.network.n_l23):
            ax[i].plot(
                self.network["statemon_N_l23"].t / ms,
                self.network["statemon_N_l23"].ge[i],
                color="r",
                label="Excitatory"
            )
            ax[i].plot(
                self.network["statemon_N_l23"].t / ms,
                self.network["statemon_N_l23"].gi[i],
                color="b",
                label="Inhibitory"
            )
            ax[i].set_ylabel(f"Synapse No.{i}")
            ax[i].set_xlabel("Time (ms)")
            # ax[i].set_ylim(-0.5, 1.5)
            ax[i].legend()
        fig.suptitle(title + subtitle)

    def draw_spike_trace(
        self,
        pre_synapse_num: list[int] = None,
        post_synapse_num: list[int] = None,
        title: str = "Spike trace",
    ):
        """
        カラム内のすべてのニューロンのスパイクトレースをプ���ットする．この関数呼び出し後にplt.show()の記述が必要．

        Args:
            neuron_num_l4 (list of int): プロットするニューロンの番号のリスト. 記述しないとすべてのニューロンをプロットする.
            neuron_num_l23 (list of int): プロットするニューロンの番号のリスト. 記述しないとすべてのニューロンをプロットする.
            title (str, optional): グラフタイトル. Defaults to "All spike trace".
        """
        # pre
        if pre_synapse_num is None:
            neurons = range(self.n_l4 * self.n_l23)
        else:
            neurons = pre_synapse_num
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id} [Pre])"
        for i, j in enumerate(neurons):
            ax[i].plot(
                self.statemon["S_l4->l23"].t / ms,
                self.statemon["S_l4->l23"].apre[j],
                color="k",
            )
            ax[i].set_ylabel(j)
            ax[i].set_xlabel("Time (ms)")
        fig.canvas.manager.set_window_title(title + subtitle)
        fig.suptitle(title + subtitle)

        # post
        if post_synapse_num is None:
            neurons = range(self.n_l4 * self.n_l23)
        else:
            neurons = post_synapse_num
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = " (column_id: %d) [Post]" % self.column_id
        for i, j in enumerate(neurons):
            ax[i].plot(
                self.statemon["S_l4->l23"].t / ms,
                self.statemon["S_l4->l23"].apost[j],
                color="k",
            )
            ax[i].set_ylabel(j)
            ax[i].set_xlabel("Time (ms)")
        fig.canvas.manager.set_window_title(title + subtitle)
        fig.suptitle(title + subtitle)

    def draw_weight_changes(
        self,
        one_fig: bool = False,
        synapse_num: list[int] = None,
        title: str = "Synapse weight",
    ):
        """
        シナプス重みをプロットする．この関数呼び出し後にplt.show()の記述が必要．
        シナプス数が多い場合，描画に時間がかかる場合があるのに注意が必要．

        Args:
            synapse_num (list of int, optional): プロットするシナプスの番号のリスト. 記述しないとすべてのシナプスをプロットする.
            title (str, optional): グラフタイトル. Defaults to "Synapse weight".
        """
        if synapse_num is None:
            neurons = range(self.n_l4 * self.n_l23)
        else:
            neurons = synapse_num
        subtitle = f" (column_id: {self.column_id}) [L4]"
        if one_fig:
            fig, ax = plt.subplots(1, 1, figsize=(12, 9))
            fig.canvas.manager.set_window_title(title + subtitle)
            for i in neurons:
                ax.plot(
                    self.statemon["S_l4->l23"].t / ms,
                    self.statemon["S_l4->l23"].w[i],
                    label="Synapse No.%d" % i,
                )
            ax.set_ylabel("Weight")
            ax.set_xlabel("Time (ms)")
            ax.set_ylim(-0.1, 1.1)
            ax.legend(loc="upper right")
            fig.suptitle(title + subtitle)
        else:
            fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
            fig.canvas.manager.set_window_title(title + subtitle)
            for i, j in enumerate(neurons):
                ax[i].plot(
                    self.statemon["S_l4->l23"].t / ms,
                    self.statemon["S_l4->l23"].w[j],
                    color="k",
                )
                ax[i].set_ylabel(f"Synapse No.{j}")
                ax[i].set_xlabel("Time (ms)")
                ax[i].set_ylim(0, 1)
            fig.suptitle(title + subtitle)

    def draw_weight(self):
        """
        すべてのシナプスの重みをヒートマップで表示し、カラーバーを追加する．
        この関数呼び出し後にplt.show()の記述が必要．
        """
        weight_mat = np.array(
            [self.S["l4->l23"].w[i] for i in range(self.n_l4 * self.n_l23)]
        ).reshape(self.n_l4, self.n_l23)
        fig = plt.figure(figsize=(12, 9))
        for img in range(self.n_l23):
            weightloop = weight_mat[:, img].reshape(
                int(np.sqrt(self.n_l4)), int(np.sqrt(self.n_l4))
            )
            ax = fig.add_subplot(
                int(np.sqrt(self.n_l23)), int(np.sqrt(self.n_l23)), img + 1
            )
            cax = ax.matshow(weightloop, cmap="Blues")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Synapse weight of L2/3 neurons")
        fig.canvas.manager.set_window_title("Synapse weight of L2/3 neurons")
        plt.colorbar(cax, ax=fig.axes, orientation="vertical", fraction=0.025, pad=0.04)

        # 初期重みをプロット
        weight_mat = np.array(
            [self.statemon["S_l4->l23"].w[i][0] for i in range(self.n_l4 * self.n_l23)]
        ).reshape(self.n_l4, self.n_l23)
        fig = plt.figure(figsize=(12, 9))
        for img in range(self.n_l23):
            weightloop = weight_mat[:, img].reshape(
                int(np.sqrt(self.n_l4)), int(np.sqrt(self.n_l4))
            )
            ax = fig.add_subplot(
                int(np.sqrt(self.n_l23)), int(np.sqrt(self.n_l23)), img + 1
            )
            cax = ax.matshow(weightloop, cmap="Blues")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Initial weight of L2/3 neurons")
        fig.canvas.manager.set_window_title("Initial weight of L2/3 neurons")
        plt.colorbar(cax, ax=fig.axes, orientation="vertical", fraction=0.025, pad=0.04)

    def draw_raster_plot(self, title="Raster plot"):
        """
        ラスタープロットをプロットする．この関数呼び出し後にplt.show()の記述が必要．

        Args:
            title (str, optional): グラフタイトル. Defaults to None.
        """
        try:
            N_inh = self.network["N_inh"]
        except KeyError:
            N_inh = None
            
        if N_inh is not None:
            fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        else:
            fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        if N_inh is not None:
            subtitle = f" (column_id: {self.column_id}) Upper[L2/3], Middle[L4], Lower[Inhibitory]"
        else:
            subtitle = f" (column_id: {self.column_id}) Upper[L2/3], Lower[L4]"
        fig.canvas.manager.set_window_title(title + subtitle)
        # L2/3のラスタープロット
        ax[0].scatter(
            self.network["spikemon_N_l23"].t / ms,
            self.network["spikemon_N_l23"].i,
            s=2,
            color="k",
            label="L2/3",
        )
        ax[0].set_ylabel("Neuron No")
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylim(-0.5, self.n_l23 + 0.5)
        ax[0].set_xlim(0, self.network.net.t / ms)
        ax[0].set_yticks(range(self.n_l23))

        ax[1].scatter(
            self.spikemon["l4"].t / ms,
            self.spikemon["l4"].i,
            s=2,
            color="k",
            label="L4",
        )
        # L4のラスタープロット
        ax[1].set_ylabel("Neuron No")
        ax[1].set_xlabel("Time (ms)")
        ax[1].set_ylim(-0.5, self.n_l4 + 0.5)
        ax[1].set_xlim(0, self.network.net.t / ms)
        ax[1].set_yticks(range(self.n_l4))
        fig.suptitle(title + subtitle)
        
        # Inhibitoryのラスタープロット
        if N_inh is not None:
            ax[2].scatter(
                self.network["spikemon_N_inh"].t / ms,
                self.network["spikemon_N_inh"].i,
                s=2,
                color="k",
                label="N_inh",
            )
            ax[2].set_xlabel("Time (ms)")
            ax[2].set_ylabel("Neuron No")
            ax[2].set_xlim(0, self.network.net.t / ms)
            ax[2].set_yticks(range(self.network.n_inh))
        
        # inputニューロンのラスタープロット
        fig= plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(
            self.spikemon["input"].t / ms,
            self.spikemon["input"].i,
            s=2,
            color="k",
            label="Input",
        )
        ax.set_xlim(0, self.network.net.t / ms)
        ax.set_yticks(range(self.network.n_inp))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron No")
        fig.suptitle("Input Neuron")
        fig.canvas.manager.set_window_title("Input Neuron")
        


    def get_firing_rate(self, simulate_duration):
        return {
            "l4": self.spikemon["l4"].count / simulate_duration,
            "l23": self.spikemon["l23"].count / simulate_duration,
            "input": self.spikemon["input"].count / simulate_duration,
        }
        
    def show_firing_rate(self, simulate_duration):
        """
        前ニューロンの発火率を表示する
        
        Args:
            simulate_duration (int): シミュレーションの実行時間。単位はsecond
        """
        print("[INFO of Firing rate]")
        rates = self.get_firing_rate(simulate_duration)
        for neuron_key in rates.keys():
            print(f"===  {neuron_key}  ===")
            for i, rate in enumerate(rates[neuron_key]):
                print(f"Neuron No.{i}: {rate}")

    def get_firing_rate_per_neuron(self):
        """
        各ニューロンの発火率を計算して返す．
        """
        # TODO 未完成
        firing_rates = {"l4": {}, "l23": {}}

        # スパイクモニターのデータを取得
        print(self.spikemon["l4"].i)
        print(self.network.t)

    def draw_firerate_map(self):
        """
        各ニューロンの発火率をマッピングしてプロットする．
        """
        # TODO 未完成
        # L4のニューロンごとの発火率を計算し、出力する
        print("spikemon", self.spikemon["l4"].num_spikes)
        print("network", self.network.t)
        fire_rates_l4 = [
            self.spikemon["l4"].num_spikes[i] / (self.network.t[-1] / second)
            for i in range(self.n_l4)
        ]
        print("L4 Fire Rates:", fire_rates_l4)

    def draw_firing_rate_changes(self):
        """
        各ニューロンの発火率の変化をプロットする．
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        title = "Firing rate changes"
        subtitle = f" (column_id: {self.column_id}) Upper[L2/3], Lower[L4]"
        fig.canvas.manager.set_window_title(title + subtitle)
        # L2/3のニューロンごとの発火率の推移をプロット
        ax[0].plot(self.network["popmon_N_l23"].t / ms, self.network["popmon_N_l23"].smooth_rate(width=50*ms))
        ax[0].set_title("L2/3 Neurons Firing Rate")
        ax[0].set_ylabel("Firing Rate (Hz)")

        # L4のニューロンごとの発火率の推移をプロット
        ax[1].plot(self.network["popmon_N_l4"].t / ms, self.network["popmon_N_l4"].smooth_rate(width=50*ms))
        ax[1].set_title("L4 Neurons Firing Rate")
        ax[1].set_ylabel("Firing Rate (Hz)")
        