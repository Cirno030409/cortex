import json
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *

defaultclock.dt = 1 * ms


class MiniColumn:
    """
    ミニカラムモデル．
    
    n_l4(int): L4層のニューロン数

    n_l23(int): L2/3層のニューロン数

    column_id(int): このカラムに割り当てる識別番号

    time_profile(TimedArray): 時間プロファイル
    
    input_neurons(PoissonGroup): 入力ニューロングループ

    synapse_between_same_layer(bool): 同じ層間のシナプスを作成するかどうか
    """

    def __init__(
        self,
        simulation_duration: int,
        n_l4: int,
        n_l23: int,
        n_inhibitory: int,
        column_id: int,
        time_profile: TimedArray = None,
        input_neurons: PoissonGroup = None,
        synapse_between_same_layer: bool = False,
        neuron_model: str = "LIF",
    ):
        self.simulation_duration = simulation_duration
        self.n_l4 = n_l4
        self.n_l23 = n_l23
        self.column_id = column_id

        # Neuron Parameters
        eqs_LIF = """
        dv/dt = ((v_rest - v) + I) / tau_m : 1
        dI/dt = -I/tau_I : 1
        v_rest : 1
        tau_m : second
        tau_I : second
        """
        # ==== Izhikevich2003モデル ====
        # a : uのスケーリング係数
        # b : vに対してuをどれくらい変化させるか
        # c : vの静止膜電位
        # d : 発火後に静止膜電位に戻るまでの時間を変化させる定数
        eqs_Izhikevich2003 = """
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I + I_noise)/ms : 1 (unless refractory)
        du/dt = (a*(b*v - u))/ms : 1
        dgsyn/dt = (-gsyn)/taugsyn : 1
        I = gsyn * (v_rev - v) : 1
        I_noise : 1
        a : 1
        b : 1
        c : 1
        d : 1
        taugsyn : second
        v_rev : 1
        """
        # dI/dt = -I/tau_I : 1
        neuron_params = {"LIF": {}, "Izhikevich2003": {}}

        #! PARAMETERS
        ## LIF
        neuron_params["LIF"]["l4"] = {
            "v_threshold_eqs": "v > -50",
            "v_threshold": -50,  # 上記の式の値のみを入力する
            "v_reset_eqs": "v = -65",
            "v_reset": -65,  # 上記の式の値のみを入力する
            "refractory": "0 * ms",
            "tau_m": 80 * ms,  # Time constant of the membrane potential
            "tau_I": 100 * ms,  # Time constant of the current
            "I": 0,
            "method": "exact",
        }
        neuron_params["LIF"]["l23"] = {
            "v_threshold_eqs": "v > -50",
            "v_threshold": -50,  # 上記の式の値のみを入力する
            "v_reset_eqs": "v = -65",
            "v_reset": -65,  # 上記の式の値のみを入力する
            "refractory": "0 * ms",
            "tau_m": 80 * ms,
            "tau_I": 100 * ms,  # Time constant of the current
            "I": 0,
            "method": "exact",
        }
        ## Izhikevich
        ### L4
        neuron_params["Izhikevich2003"]["l4"] = {
            "v_threshold_eqs": "v > -50",
            "v_threshold": -50,  # 上記の式の値のみを入力する
            "v_reset_eqs": "v=c; u+=d",
            "v_reset": -65,  # 上記の式の値のみを入力する
            "v_rev": 0,
            "refractory": "3 * ms",
            "neuron_type": "RS",
            "I_noise": 0,  # ノイズ入力(自発発火用)
            "tau_I": 100 * ms,  # Time constant of the current
            "tau_gsyn": 2 * ms,  # コンダクタンスの時定数
            "tau_gsyn": 1 * ms,  # Time constant of the synapse
            "v_reversal": 0,  # Reversal potential
            "method": "euler",
        }
        ### L2/3
        neuron_params["Izhikevich2003"]["l23"] = {
            "v_threshold_eqs": "v > -50",
            "v_threshold": -50,  # 上記の式の値のみを入力する
            "v_reset_eqs": "v=c; u+=d",
            "v_reset": -65,  # 上記の式の値のみを入力する
            "v_rev": 0,
            "refractory": "3 * ms",
            "neuron_type": "RS",
            "I_noise": 0,  # ノイズ入力(自発発火用)
            "tau_I": 80 * ms,  # Time constant of the current
            "tau_gsyn": 1 * ms,  # Time constant of the synapse
            "method": "euler",
        }
        ### Inhibitory
        neuron_params["Izhikevich2003"]["inhibitory"] = {
            "v_threshold_eqs": "v > -50",
            "v_threshold": -50,  # 上記の式の値のみを入力する
            "v_reset_eqs": "v=c; u+=d",
            "v_reset": -65,  # 上記の式の値のみを入力する
            "v_rev": -60,
            "refractory": "3 * ms",
            "neuron_type": "RS",
            "I_noise": 0,  # ノイズ入力(自発発火用)
            "tau_I": 80 * ms,  # Time constant of the current
            "tau_syn": 50 * ms,  # Time constant of the synapse
            "tau_gsyn": 1 * ms,  # Time constant of the synapse
            "method": "euler",
        }
        ## STDP
        stdp_params = {
            "wmax": 1,  # 最大重み
            "alpha": 0.01,  # スパイクトレースの収束地点
            "tau_pre": 20 * ms,  # 前ニューロンのスパイクトレースの時定数
            "tau_post": 20 * ms,  # 後ニューロンのスパイクトレースの時定数
            "Apre": 0.01,  # 前ニューロンのスパイクトレースのリセット値
            "Apost": 1,  # 後ニューロンのスパイクトレースのリセット値
        }
        ## Synapse models (of "input->l4", "l4->inhibitory", "inhibitory->l23")
        synapse_model_params = {"input->l4": {}, "l4->inhibitory": {}, "inhibitory->l23": {}}
        ### input->Layer4
        synapse_model_params["input->l4"] = {
            "tau_post": 20 * ms,  # 後ニューロンのスパイクトレースの時定数
            "tau_pre": 20 * ms,  # 前ニューロンのスパイクトレースの時定数
            "tau_syn": 50 * ms,  # シナプストレースの時定数
            "tau_gsyn": 1 * ms,  # コンダクタンスの時定数
            "v_reversal": 0,  # Reversal potential
        }
        ### Layer4 -> Layer2/3
        synapse_model_params["l4->l23"] = {
            "tau_post": 20 * ms,  # 後ニューロンのスパイクトレースの時定数
            "tau_pre": 20 * ms,  # 前ニューロンのスパイクトレースの時定数
            "tau_gsyn": 1 * ms,  # コンダクタンスの時定数
            "v_reversal": 0,  # Reversal potential
        }
        ### Layer2/3 -> Inhibitory
        synapse_model_params["l23->inhibitory"] = {
            "tau_post": 20 * ms,  # 後ニューロンのスパイクトレースの時定数
            "tau_pre": 20 * ms,  # 前ニューロンのスパイクトレースの時定数
            "tau_gsyn": 1 * ms,  # コンダクタンスの時定数
            "v_reversal": 0,  # Reversal potential
        }
        ### Inhibitory -> Layer2/3
        synapse_model_params["inhibitory->l23"] = {
            "tau_post": 20 * ms,  # 後ニューロンのスパイクトレースの時定数
            "tau_pre": 20 * ms,  # 前ニューロンのスパイクトレースの時定数
            "tau_gsyn": 1 * ms,  # コンダクタンスの時定数
            "v_reversal": -60,  # Reversal potential
        }

        # Synapse equations
        ## synapse eqs without STDP
        syn_eqs = """
        w : 1
        """
        syn_eqs_on_pre = """
        gsyn += w
        """
        ## STDP
        eqs_stdp = """
        w : 1
        Apre : 1
        Apost : 1
        taupost : second
        taupre : second
        wmax : 1
        alpha : 1
        dapre/dt = (-apre - alpha)/taupre : 1 (clock-driven)
        dapost/dt = (-apost)/taupost : 1 (clock-driven)

        """
        eqs_stdp_on_pre = """
        gsyn_post += w
        apre = Apre
        w = clip(w + apost, 0, wmax)
        """
        eqs_stdp_on_post = """
        apost = Apost
        w = clip(w + apre, 0, wmax)
        """
        # In brian2,
        # v_post : voltage of the post-synaptic neuron
        # I_post : current of the post-synaptic neuron
        # _post, _preは予約語

        self.N = {}
        self.S = {}
        self.spikemon = {}
        self.statemon = {}

        #! NEURON SETTINGS ##############################################################################################
        if neuron_model == "LIF":
            eqs_neuron_model = eqs_LIF
        elif neuron_model == "Izhikevich2003":
            eqs_neuron_model = eqs_Izhikevich2003
            with open("Izhikevich2003_parameters.json") as f:
                izhikevich_params = json.load(f)
        else:
            raise ValueError(
                "Invalid neuron model: %s. Please choose 'LIF' or 'Izhikevich2003'."
                % neuron_model
            )
        ## Neuron group of INPUT (初期化)
        self.N["input"] = input_neurons
        ## Neuron group of L4
        self.N["l4"] = NeuronGroup(
            n_l4,
            eqs_neuron_model,
            threshold=neuron_params[neuron_model]["l4"]["v_threshold_eqs"],
            reset=neuron_params[neuron_model]["l4"]["v_reset_eqs"],
            refractory=neuron_params[neuron_model]["l4"]["refractory"],
            method=neuron_params[neuron_model]["l4"]["method"],
        )

        ## Neuron group of L2/3
        self.N["l23"] = NeuronGroup(
            n_l23,
            eqs_neuron_model,
            threshold=neuron_params[neuron_model]["l23"]["v_threshold_eqs"],
            reset=neuron_params[neuron_model]["l23"]["v_reset_eqs"],
            refractory=neuron_params[neuron_model]["l23"]["refractory"],
            method=neuron_params[neuron_model]["l23"]["method"],
        )
        
        ## Neuron group of Inhibitory
        # self.N["inhibitory"] = NeuronGroup(
        #     n_inhibitory,
        #     eqs_neuron_model,
        #     threshold=neuron_params[neuron_model]["inhibitory"]["v_threshold_eqs"],
        #     reset=neuron_params[neuron_model]["inhibitory"]["v_reset_eqs"],
        #     refractory=neuron_params[neuron_model]["inhibitory"]["refractory"],
        #     method=neuron_params[neuron_model]["inhibitory"]["method"],
        # )

        ## Set the parameters of the neurons
        ### 共通しているパラメータ
        self.N["l4"].v = neuron_params[neuron_model]["l4"]["v_reset"]
        self.N["l4"].I_noise = neuron_params[neuron_model]["l4"]["I_noise"]

        self.N["l23"].v = neuron_params[neuron_model]["l23"]["v_reset"]
        self.N["l23"].I_noise = neuron_params[neuron_model]["l23"]["I_noise"]
        
        # self.N["inhibitory"].v = neuron_params[neuron_model]["inhibitory"]["v_reset"]
        # self.N["inhibitory"].tau_I = neuron_params[neuron_model]["inhibitory"]["tau_I"]
        # self.N["inhibitory"].I_noise = neuron_params[neuron_model]["inhibitory"]["I_noise"]

        ### モデルごとに異なるパラメータ
        if neuron_model == "LIF":
            self.N["l4"].v_rest = neuron_params["LIF"]["l4"]["v_reset"]
            self.N["l23"].v_rest = neuron_params["LIF"]["l23"]["v_reset"]
            self.N["l4"].tau_m = neuron_params["LIF"]["l4"]["tau_m"]
            self.N["l23"].tau_m = neuron_params["LIF"]["l23"]["tau_m"]
        elif neuron_model == "Izhikevich2003":
            for neuron_group in self.N.keys():
                if neuron_group == "input":
                    continue
                # Neuron type parameters
                self.N[neuron_group].a = izhikevich_params[
                    neuron_params["Izhikevich2003"][neuron_group]["neuron_type"]
                ]["a"]
                self.N[neuron_group].b = izhikevich_params[
                    neuron_params["Izhikevich2003"][neuron_group]["neuron_type"]
                ]["b"]
                self.N[neuron_group].c = izhikevich_params[
                    neuron_params["Izhikevich2003"][neuron_group]["neuron_type"]
                ]["c"]
                self.N[neuron_group].d = izhikevich_params[
                    neuron_params["Izhikevich2003"][neuron_group]["neuron_type"]
                ]["d"]
                # Others
                self.N[neuron_group].v_rev = neuron_params["Izhikevich2003"][neuron_group]["v_rev"]
                self.N[neuron_group].taugsyn = neuron_params["Izhikevich2003"][neuron_group]["tau_gsyn"]
                

        #! SYNAPSE SETTINGS ##############################################################################################
        ## Build Synapses
        ### INPUT -> Layer4
        self.S["input->l4"] = Synapses(
            self.N["input"],
            self.N["l4"],
            model=syn_eqs,
            on_pre=syn_eqs_on_pre,
            delay=1 * ms,
            method="euler",
        )
        self.S["input->l4"].connect("i == j")
        ### Layer4 -> Layer2/3
        self.S["l4->l23"] = Synapses(
            self.N["l4"],
            self.N["l23"],
            model=eqs_stdp,
            on_pre=eqs_stdp_on_pre,
            on_post=eqs_stdp_on_post,
            delay=1 * ms,
            method="euler",
        )
        self.S["l4->l23"].connect()
        ### Layer2/3 -> Inhibitory
        # self.S["l23->inhibitory"] = Synapses(
        #     self.N["l23"],
        #     self.N["inhibitory"],
        #     model=syn_eqs,
        #     on_pre=syn_eqs_on_pre,
        #     delay=1 * ms,
        #     method="euler",
        # )
        # self.S["l23->inhibitory"].connect("i == j")
        # ### Layer2/3 <- Inhibitory
        # self.S["inhibitory->l23"] = Synapses(
        #     self.N["inhibitory"],
        #     self.N["l23"],
        #     model=syn_eqs,
        #     on_pre=syn_eqs_on_pre,
        #     delay=1 * ms,
        #     method="euler",
        # )
        # self.S["inhibitory->l23"].connect("i != j")

        # シナプス重みの最大値と最小値の定義
        ## Synapse settings
        w_max = 1.0
        w_min = 0.0

        ## 全シナプスモデルに対してのパラメータ代入
        for synapse_key in self.S:
            if synapse_key == "l4->l23": # 学習を行うシナプス
                # STDP固有パラメータ
                self.S[synapse_key].w = "rand() * (w_max - w_min) + w_min"
                self.S[synapse_key].wmax = stdp_params["wmax"]
                self.S[synapse_key].alpha = stdp_params["alpha"]
                self.S[synapse_key].Apre = stdp_params["Apre"]
                self.S[synapse_key].Apost = stdp_params["Apost"]
                self.S[synapse_key].taupre = stdp_params["tau_pre"]
                self.S[synapse_key].taupost = stdp_params["tau_post"]
                
            elif synapse_key == "input->l4" or synapse_key == "l23->inhibitory" or synapse_key == "inhibitory->l23":
                self.S[synapse_key].w = 1.0 # 重みの固定
                
            # ### ニューロングループ別のシナプスモデルパラメータを代入
            # self.S[synapse_key].v_rev = synapse_model_params[synapse_key]["v_reversal"]
            # self.S[synapse_key].tausyn = synapse_model_params[synapse_key]["tau_syn"]
            # self.S[synapse_key].syn_init = synapse_model_params[synapse_key]["syn_init"]
            # self.S[synapse_key].taugsyn = synapse_model_params[synapse_key]["tau_gsyn"]


        # Time profileで刺激を与える場合
        if time_profile is not None:
            self.N["l4"].run_regularly("I = time_profile(t)")

        #! MONITOR SETTINGS ##############################################################################################
        for neuron_key in self.N.keys(): # すべてのニューロングループのState Monitorを作成
            if neuron_key == "input":
                self.spikemon[neuron_key] = SpikeMonitor(self.N[neuron_key])

            else:
                self.spikemon[neuron_key] = SpikeMonitor(self.N[neuron_key])
                self.statemon[neuron_key] = StateMonitor(
                    self.N[neuron_key], ["v", "I"], record=True
                )

        self.statemon["S_l4->l23"] = StateMonitor(
            self.S["l4->l23"],
            ["w", "apre", "apost", "gsyn"],
            record=True,
        )
        

        self.network = Network(self.N, self.S, self.spikemon, self.statemon)
        
        # Save Parameters for plotting
        self.v_rest_for_plot_l4 = neuron_params[neuron_model]["l4"]["v_reset"]
        self.v_th_for_plot_l4 = neuron_params[neuron_model]["l4"]["v_threshold"]
        self.v_rest_for_plot_l23 = neuron_params[neuron_model]["l23"]["v_reset"]
        self.v_th_for_plot_l23 = neuron_params[neuron_model]["l23"]["v_threshold"]

    def run(self, duration):
        self.network.run(duration)

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
            ax[i].plot(self.statemon["l4"].t / ms, self.statemon["l4"].I[j], color="k")
            ax[i].set_ylabel(f"Neuron No.{j}")
            # ax[i].set_ylim(0, 120)
            ax[i].set_xlabel("Time (ms)")
        fig.suptitle(title)

        if neuron_num_l23 is None:
            neurons = range(self.n_l23)
        else:
            neurons = neuron_num_l23
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id}) [L2/3]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i, j in enumerate(neurons):
            ax[i].plot(
                self.statemon["l23"].t / ms, self.statemon["l23"].I[j], color="k"
            )
            ax[i].set_ylabel(f"Neuron No.{j}")
            # ax[i].set_ylim(0, 120)
            ax[i].set_xlabel("Time (ms)")
        fig.suptitle(title)

    def draw_conductance(
        self, synapse_num: list[int] = None, title: str = "Conductance"
    ):
        """
        シナプスモデルのコンダクタンスのグラフを描画する．
        """
        if synapse_num is None:
            neurons = range(self.n_l4 * self.n_l23)
        else:
            neurons = synapse_num
        fig, ax = plt.subplots(len(neurons) + 1, 1, sharex=True, figsize=(12, 9))
        subtitle = f" (column_id: {self.column_id}) [L4->L2/3]"
        fig.canvas.manager.set_window_title(title + subtitle)
        for i, j in enumerate(neurons):
            ax[i].plot(
                self.statemon["S_l4->l23"].t / ms,
                self.statemon["S_l4->l23"].gsyn[j],
                color="k",
            )
            ax[i].set_ylabel(f"Synapse No.{j}")
            ax[i].set_xlabel("Time (ms)")
            # ax[i].set_ylim(-0.5, 1.5)
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
        fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        subtitle = f" (column_id: {self.column_id}) Upper[L2/3], Lower[L4]"
        fig.canvas.manager.set_window_title(title + subtitle)
        ax[0].scatter(
            self.spikemon["l23"].t / ms,
            self.spikemon["l23"].i,
            s=2,
            color="k",
            label="L2/3",
        )
        ax[0].set_ylabel("Neuron No")
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylim(-0.5, self.n_l23 + 0.5)
        ax[0].set_xlim(0, self.simulation_duration / ms)
        ax[0].set_yticks(range(self.n_l23))

        ax[1].scatter(
            self.spikemon["l4"].t / ms,
            self.spikemon["l4"].i,
            s=2,
            color="k",
            label="L4",
        )
        ax[1].set_ylabel("Neuron No")
        ax[1].set_xlabel("Time (ms)")
        ax[1].set_ylim(-0.5, self.n_l4 + 0.5)
        ax[1].set_xlim(0, self.simulation_duration / ms)
        ax[1].set_yticks(range(self.n_l4))
        fig.suptitle(title + subtitle)

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


if __name__ == "__main__":
    print("This is a module for MiniColumn. You can't run this file directly.")
