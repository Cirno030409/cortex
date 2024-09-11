import pprint

from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from brian2 import *


class Diehl_and_Cook_WTA:
    """
    Diehl and CookのWTAネットワークを作成します。
    
    Args:
        n_inp (int): 入力ニューロンの数\n
        n_e (int): 興奮ニューロンの数\n
        n_i (int): 抑制ニューロンの数\n
        max_rate (float): 最大発火率\n
        neuron_params_e (dict): 興奮ニューロンのパラメータ\n
        neuron_params_i (dict): 抑制ニューロンのパラメータ\n
        static_synapse_params_ei (dict): 興奮ニューロンから抑制ニューロンへのシナプスのパラメータ\n
        static_synapse_params_ie (dict): 抑制ニューロンから興奮ニューロンへのシナプスのパラメータ\n
        stdp_synapse_params (dict): STDPシナプスのパラメータ
    Returns:
        brian2.Network: ネットワーク\n
        list: ニューロンリスト\n
        list: シナプスリスト
    Methods:
        enable_learning(): 学習を有効にします。\n
        disable_learning(): 学習を無効にします。\n
        change_image(image:np.ndarray, spontaneous_rate:int=0): 入力画像を変更します。
    """
    def __init__(self, enable_monitor:bool, n_inp, n_e, n_i, max_rate, neuron_params_e, neuron_params_i, static_synapse_params_ei, static_synapse_params_ie, stdp_synapse_params):
        # Make instances of neurons and synapses
        neuron_e = Conductance_LIF(neuron_params_e)
        neuron_i = Conductance_LIF(neuron_params_i)
        self.neuron_inp = Poisson_Input()
        synapse_ei = NonSTDP(static_synapse_params_ei)
        synapse_ie = NonSTDP(static_synapse_params_ie)
        synapse_stdp = STDP(stdp_synapse_params)
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_0"] = self.neuron_inp(n_inp, max_rate=max_rate, name="N_0")
        self.obj["N_1"] = neuron_e(n_e, "N_middle", name="N_1")
        self.obj["N_2"] = neuron_i(n_i, "N_output", name="N_2")

        self.obj["S_0"] = synapse_stdp(self.obj["N_0"], self.obj["N_1"], name="S_0", connect=True) # 入力層から興奮ニューロン
        self.obj["S_1"] = synapse_ei(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", delay=0*ms, connect="i==j") # 興奮ニューロンから抑制ニューロン
        self.obj["S_2"] = synapse_ie(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", delay=0*ms, connect="i!=j") # 側抑制
        
        # Create monitors
        if enable_monitor:
            self.obj["spikemon_0"] = SpikeMonitor(self.obj["N_0"], record=True, name="spikemon_0")
            self.obj["spikemon_2"] = SpikeMonitor(self.obj["N_2"], record=True, name="spikemon_2")
            self.obj["statemon_1"] = StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_1")
            self.obj["statemon_2"] = StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_2")
            self.obj["statemon_S"] = StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_S")
        self.obj["spikemon_1"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_1") # ラベル割当に必要

        self.network = Network(self.obj.values()) # ネットワークを作成

    def enable_learning(self):
        """
        学習を有効にします。
        """
        self.obj["S_0"].namespace["sw"] = 1
        
    def disable_learning(self):
        """
        学習を無効にします。
        """
        self.obj["S_0"].namespace["sw"] = 0
        
    def change_image(self, image:np.ndarray, spontaneous_rate:int=0):
        """
        入力画像を変更します。

        Args:
            image (np.ndarray): 入力画像\n
            spontaneous_rate (int, optional): 自発発火率. Defaults to 0.
        """
        self.neuron_inp.change_image(image, spontaneous_rate)

class Cortex:
    """
    複数のミニカラムが接続した，新皮質ネットワークを作成します。
    """
    def __init__(self):
        self.net = Network()
        self.minicolumns = []
        
    def add_minicolumns(self, num:int, n_inp:int, n_l4:int, n_l23:int, n_inh:int, monitors:bool):
        """
        ミニカラムを作成して，ネットワークに追加します。

        Args:
            n_inp (int): 入力ニューロンの数
            n_l4 (int): L4ニューロンの数
            n_l23 (int): L2/3ニューロンの数
            n_inh (int): インハリッチニューロンの数
            monitors (bool): モニターを作成するかどうか

        Returns:
            mini_column(brian2.Network): １つのミニカラムネットワーク
        """
        for i in range(num):
            self.minicolumns.append(MiniColumn("LIF", i, n_inp, n_l4, n_l23, n_inh, monitors))
            print("[Added mini-columns:", i, "]")
            
    def connect_minicolumns(self, i:int, j:int, condition:True):
        NonSTDP = MySynapses.NonSTDP()
        synapse = NonSTDP(self.minicolumns[i].obj["N_l23"], self.minicolumns[j].obj["N_l4"], exc_or_inh="exc", connect=condition)
        self.net.add(synapse)
        print("[Connected mini-columns:", i, "and", j, "]")
        
    def run(self, inp:np.ndarray, max_rate:float, duration:float):
        for i in range(len(inp)):
            inp[i] = inp[i] / 255.0
            inp[i] = inp[i] * max_rate
        self.minicolumns[0].obj["N_input"].rates = inp * Hz
        for minicolumn in self.minicolumns:
            self.net.add(minicolumn.make_column())
        print("[Running Cortex ...]")
        pprint.pprint(self.net.objects)
        self.net.run(duration)
        print("[Finished!]")

class MiniColumn:
    """
    ミニカラムネットワークを作成します。
    """
    def __init__(self, model:str, column_id:int, n_inp:int, n_l4:int, n_l23:int, n_inh:int = 0, monitors:bool = False):
        self.id = column_id
        self.n_l4 = n_l4
        self.n_l23 = n_l23
        self.n_inp = n_inp
        self.n_inh = n_inh
        
        self.network = Network()
        
        if model == "Izhikevich":
            neuron = Neurons.Conductance_Izhikevich2003(neuron_type="RS")
        elif model == "LIF":
            neuron = Neurons.Conductance_LIF()
        else:
            raise Exception("Invalid neuron model: " + model)
        stdp_synapse = MySynapses.STDP()
        nonstdp_synapse = MySynapses.NonSTDP()
        
        # Make NeuronGroups
        self.obj = {}
        self.obj["N_input"] = PoissonGroup(n_inp, rates=np.zeros(n_inp) * Hz)
        self.obj["N_l4"] = neuron(n_l4, exc_or_inh="exc", name=f"N_l4_{self.id}")
        self.obj["N_l23"] = neuron(n_l23, exc_or_inh="exc", name=f"N_l23_{self.id}")

        ## Connect Synapse
        self.obj["S_input_l4"] = nonstdp_synapse(self.obj["N_input"], self.obj["N_l4"], exc_or_inh="exc", connect="i==j", name=f"S_input_l4_{self.id}")
        self.obj["S_l4_l23"] = stdp_synapse(self.obj["N_l4"], self.obj["N_l23"], connect=True, name=f"S_l4_l23_{self.id}")
        
        if n_inh != 0:
            self.obj["N_inh"] = neuron(n_inh, "inh", name=f"N_inh_{self.id}")
            self.obj["S_l23_inh"] = nonstdp_synapse(self.obj["N_l23"], self.obj["N_inh"], exc_or_inh="exc", connect="i==j", name=f"S_l23_inh_{self.id}")
            self.obj["S_inh_l23"] = nonstdp_synapse(self.obj["N_inh"], self.obj["N_l23"], exc_or_inh="inh", w=2, connect="i!=j", name=f"S_inh_l23_{self.id}")
        
        # Make Monitors
        if monitors:
            for i in list(self.obj.keys()):
                if i == "S_l4_l23": # STDP用Monitor
                    self.obj["statemon_"+i] = StateMonitor(self.obj[i], ["w", "apre", "apost"], record=True, name=f"statemon_{i}_{self.id}")
                elif i == "N_l4" or i == "N_l23" or i == "N_inh":
                    self.obj["spikemon_"+i] = SpikeMonitor(self.obj[i], record=True, name=f"spikemon_{i}_{self.id}")
                    self.obj["statemon_"+i] = StateMonitor(self.obj[i], ["v", "Ie", "Ii", "ge", "gi", "theta"], record=True, name=f"statemon_{i}_{self.id}")
                    self.obj["popmon_"+i] = PopulationRateMonitor(self.obj[i], name=f"popmon_{i}_{self.id}")
                elif i == "N_input": # PoissonGroup用Monitor
                    self.obj["spikemon_"+i] = SpikeMonitor(self.obj[i], record=True, name=f"spikemon_{i}_{self.id}")
        
    def __getitem__(self, key):
        return self.network[key]
    
    def __setitem__(self, key, value):
        self.network[key] = value
        
    
    def make_column(self):
        """
        一つのミニカラムを作成します。

        Returns:
            brian2.Network: 一つのミニカラムのネットワーク
        """
        net = Network(self.obj.values())
        # net.run(0 * second)
        return net
        
    def connect_minicolumn(self, post:Network, condition:True):
        """
        postミニカラムをこのミニカラムに対して接続します。
        このミニカラムのN_l23とpostミニカラムのN_l4を接続します。
        
        Args:
            post (MiniColumn): このミニカラムと接続するpostミニカラム
        """
        nonstdp_synapse = MySynapses.NonSTDP()
        synapse = nonstdp_synapse(self.obj["N_l23"], post["N_l4"], "exc", connect=condition)
        # self.network.add(synapse)
        
    def run(self, inp:np.ndarray, max_rate:float, duration:float):
        """
        ネットワークを実行します。

        Args:
            inp (np.ndarray): ネットワークに入力する0~255の画素値リストを入力します。
            max_rate (float): 最大発火率を指定します。
            duration (float): ネットワークの実行時間を指定します。
        """
        for i in range(len(inp)):
            inp[i] = inp[i] / 255.0
            inp[i] = inp[i] * max_rate
        self.obj["N_input"].rates = inp * Hz
        self.network.run(duration)
        
    def reset(self):
        default_neuron_param = {
            "v" : -65,
            "ge" : 0,
            "gi" : 0
        }
        default_synapse_param = {
            "apre" : 0,
            "apost" : 0
        }
        net_names = [obj.name for obj in self.network.objects if hasattr(obj, 'name')]
        for i in net_names:
            if i.startswith("N_") and i != "N_input":
                self.network[i].set_states(default_neuron_param)
            elif i == "S_l4_l23":
                self.network[i].set_states(default_synapse_param)
        self.network.run(0 * second)
        







