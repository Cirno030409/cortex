import pprint

from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from brian2 import *
import Brian2_Framework.Tools as tools



class Diehl_and_Cook_WTA:
    """
    Diehl and CookのWTAネットワークを作成します。
    
    Args:
        enable_monitor (bool): モニタリングを有効にするか\n
        params_json_path (str): パラメータを保存したJSONファイルのパス
    Returns:
        brian2.Network: ネットワーク\n
        list: ニューロンリスト\n
        list: シナプスリスト
    Methods:
        enable_learning(): 学習を有効にします。\n
        disable_learning(): 学習を無効にします。\n
        change_image(image:np.ndarray, spontaneous_rate:int=0): 入力画像を変更します。
    """
    def __init__(self, enable_monitor:bool, params_json_path:str):
        # Make instances of neurons and synapses
        params = tools.load_parameters(params_json_path)
        neuron_e = Conductance_LIF(params["neuron_params_e"])
        neuron_i = Conductance_LIF(params["neuron_params_i"])
        self.neuron_inp = Poisson_Input()
        synapse_ei = NonSTDP(params["static_synapse_params_ei"])
        synapse_ie = NonSTDP(params["static_synapse_params_ie"])
        synapse_stdp = STDP(params["stdp_synapse_params"])
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_0"] = self.neuron_inp(params["n_inp"], max_rate=params["max_rate"], name="N_0")
        self.obj["N_1"] = neuron_e(params["n_e"], name="N_1")
        self.obj["N_2"] = neuron_i(params["n_i"], name="N_2")

        self.obj["S_0"] = synapse_stdp(self.obj["N_0"], self.obj["N_1"], name="S_0", connect=True) # 入力層から興奮ニューロン
        self.obj["S_1"] = synapse_ei(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", delay=0*ms, connect="i==j") # 興奮ニューロンから抑制ニューロン
        self.obj["S_2"] = synapse_ie(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", delay=0*ms, connect="i!=j") # 側抑制
        
        # Create monitors
        if enable_monitor:
            self.obj["spikemon_0"] = SpikeMonitor(self.obj["N_0"], record=True, name="spikemon_0")
            self.obj["spikemon_1"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_1")
            self.obj["spikemon_2"] = SpikeMonitor(self.obj["N_2"], record=True, name="spikemon_2")
            self.obj["statemon_1"] = StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_1")
            self.obj["statemon_2"] = StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_2")
            self.obj["statemon_S"] = StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_S")
        self.obj["spikemon_for_assign"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_for_assign") # ラベル割当に必要

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
        
class Chunk_WTA:
    """
    チャンク画像用ネットワークを作成します。
    
    Args:
        enable_monitor (bool): モニタリングを有効にするか\n
        params_json_path (str): パラメータを保存したJSONファイルのパス

    Methods:
        enable_learning(): 学習を有効にします。\n
        disable_learning(): 学習を無効にします。\n
        change_image(image:np.ndarray, spontaneous_rate:int=0): 入力画像を変更します。
    """
    def __init__(self, enable_monitor:bool, params_json_path:str):
        # Make instances of neurons and synapses
        params = tools.load_parameters(params_json_path)
        neuron_1e = Conductance_LIF(params["neuron_params_1e"])
        neuron_1i = Conductance_LIF(params["neuron_params_1i"])
        neuron_2e = Conductance_LIF(params["neuron_params_2e"])
        neuron_2i = Conductance_LIF(params["neuron_params_2i"])
        self.neuron_inp = Poisson_Input()
        synapse_1ei = NonSTDP(params["static_synapse_params_1ei"])
        synapse_1ie = NonSTDP(params["static_synapse_params_1ie"])
        synapse_2ei = NonSTDP(params["static_synapse_params_2ei"])
        synapse_2ie = NonSTDP(params["static_synapse_params_2ie"])
        synapse_1stdp = STDP(params["stdp_synapse_params_1"])
        synapse_2stdp = STDP(params["stdp_synapse_params_2"])
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        ## NeuronGroups
        self.obj["N_input"] = self.neuron_inp(params["n_inp"], max_rate=params["max_rate"], name="N_input")
        self.obj["N_1_exc"] = neuron_1e(params["n_e"], name="N_1_exc")
        self.obj["N_1_inh"] = neuron_1i(params["n_i"], name="N_1_inh")
        self.obj["N_2_exc"] = neuron_2e(params["n_e"], name="N_2_exc")
        self.obj["N_2_inh"] = neuron_2i(params["n_i"], name="N_2_inh")

        ## Synapses
        self.obj["S_0"] = synapse_1stdp(self.obj["N_input"], self.obj["N_1_exc"], name="S_0", connect=True) # 入力層から興奮ニューロン
        self.obj["S_1_ei"] = synapse_1ei(self.obj["N_1_exc"], self.obj["N_1_inh"], "exc", name="S_1_ei", delay=0*ms, connect="i==j") # 興奮ニューロンから抑制ニューロン
        self.obj["S_1_ie"] = synapse_1ie(self.obj["N_1_inh"], self.obj["N_1_exc"], "inh", name="S_1_ie", delay=0*ms, connect="i!=j") # 側抑制
        self.obj["S_1_2"] = synapse_2stdp(self.obj["N_1_exc"], self.obj["N_2_exc"], name="S_1_2", connect=True) # １層から２層目の接続
        self.obj["S_2_ei"] = synapse_2ei(self.obj["N_2_exc"], self.obj["N_2_inh"], "exc", name="S_2_ei", delay=0*ms, connect="i==j") # 興奮ニューロンから抑制ニューロン
        self.obj["S_2_ie"] = synapse_2ie(self.obj["N_2_inh"], self.obj["N_2_exc"], "inh", name="S_2_ie", delay=0*ms, connect="i!=j") # 側抑制
        
        # Create monitors
        if enable_monitor:
            self.obj["spikemon_input"] = SpikeMonitor(self.obj["N_input"], record=True, name="spikemon_input")
            self.obj["spikemon_1_exc"] = SpikeMonitor(self.obj["N_1_exc"], record=True, name="spikemon_1_exc")
            self.obj["statemon_1stdp"] = StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_1stdp")
            self.obj["statemon_2stdp"] = StateMonitor(self.obj["S_1_2"], ["w", "apre", "apost"], record=0, name="statemon_2stdp")
            self.obj["statemon_N_1_exc"] = StateMonitor(self.obj["N_1_exc"], ["v", "Ie", "Ii", "ge", "gi"], record=50, name="statemon_N_1_exc")
        self.obj["spikemon_2_exc"] = SpikeMonitor(self.obj["N_2_exc"], record=True, name="spikemon_2_exc") # ラベル割当に必要

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

class Center_Surround_WTA:
    """
    抑制をCenter-Surroundで行うWTAネットワーク。
    """
    def __init__(self, enable_monitor:bool, params_json_path:str):
        # Make instances of neurons and synapses
        params = tools.load_parameters(params_json_path)
        neuron_e = Conductance_LIF(params["neuron_params_e"])
        neuron_i = Conductance_LIF(params["neuron_params_i"])
        self.neuron_inp = Poisson_Input()
        synapse_ei = NonSTDP(params["static_synapse_params_ei"])
        synapse_ie = NonSTDP(params["static_synapse_params_ie"])
        synapse_stdp = STDP(params["stdp_synapse_params"])
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_0"] = self.neuron_inp(params["n_inp"], max_rate=params["max_rate"], name="N_0")
        self.obj["N_1"] = neuron_e(params["n_e"], name="N_1")
        self.obj["N_2"] = neuron_i(params["n_i"], name="N_2")

        self.obj["S_0"] = synapse_stdp(self.obj["N_0"], self.obj["N_1"], name="S_0", connect=True) # 入力層から興奮ニューロン
        self.obj["S_1"] = synapse_ei(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", delay=0*ms, connect="i==j") # 興奮ニューロンから抑制ニューロン
        k = params["static_synapse_params_ie"]["k"]
        self.obj["S_2"] = synapse_ie(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", delay=0*ms, connect=f"i!=j and abs(i-j)>{k}") # Center-Surround抑制
        
        # Create monitors
        if enable_monitor:
            self.obj["spikemon_0"] = SpikeMonitor(self.obj["N_0"], record=True, name="spikemon_0")
            self.obj["spikemon_1"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_1")
            self.obj["spikemon_2"] = SpikeMonitor(self.obj["N_2"], record=True, name="spikemon_2")
            self.obj["statemon_1"] = StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_1")
            self.obj["statemon_2"] = StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_2")
            self.obj["statemon_S"] = StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_S")
        self.obj["spikemon_for_assign"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_for_assign") # ラベル割当に必要

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






