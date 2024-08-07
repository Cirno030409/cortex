import matplotlib.pyplot as plt
import numpy as np
from brian2.units import *


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
        カラム内のすべてのニューロンのスパイクトレースをプロットする．この関数呼び出し後にplt.show()の記述が必要．

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
        