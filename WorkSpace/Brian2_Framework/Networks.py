import pprint

from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from brian2 import *
import Brian2_Framework.Tools as tools


class Network_Frame:
    """
    ネットワークに一律の機能を提供する親クラス
    
    Methods:
        enable_learning(): 学習を有効にします。\n
        disable_learning(): 学習を無効にします。\n
        change_image(image:np.ndarray, spontaneous_rate:int=0): 入力画像を変更します。
    """
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
        入力画像を変更します。obj["N_0"]が入力層であると想定しています。

        Args:
            image (np.ndarray): 入力画像\n
            spontaneous_rate (int, optional): 自発発火率. Defaults to 0.
        """
        self.obj["N_0"].change_image(image, spontaneous_rate)
        
class Diehl_and_Cook_WTA(Network_Frame):
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
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_0"] = Poisson_Input(params["n_inp"], max_rate=params["max_rate"], name="N_0")
        self.obj["N_1"] = Conductance_LIF(params["n_e"], params["neuron_params_e"], name="N_1")
        self.obj["N_2"] = Conductance_LIF(params["n_i"], params["neuron_params_i"], name="N_2")

        self.obj["S_0"] = STDP_Synapse(self.obj["N_0"], self.obj["N_1"], name="S_0", connect=True, params=params["stdp_synapse_params"]) # 入力層から興奮ニューロン
        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", delay=0*ms, connect="i==j", params=params["static_synapse_params_ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", delay=0*ms, connect="i!=j", params=params["static_synapse_params_ie"]) # 側抑制
        
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
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        ## NeuronGroups
        self.obj["N_input"] = self.neuron_inp(params["n_inp"], max_rate=params["max_rate"], name="N_input")
        self.obj["N_1_exc"] = Conductance_LIF(params["n_e"], params["neuron_params_1e"], name="N_1_exc")
        self.obj["N_1_inh"] = Conductance_LIF(params["n_i"], params["neuron_params_1i"], name="N_1_inh")
        self.obj["N_2_exc"] = Conductance_LIF(params["n_e"], params["neuron_params_2e"], name="N_2_exc")
        self.obj["N_2_inh"] = Conductance_LIF(params["n_i"], params["neuron_params_2i"], name="N_2_inh")

        ## Synapses
        self.obj["S_0"] = STDP_Synapse(self.obj["N_input"], self.obj["N_1_exc"], name="S_0", connect=True, params=params["stdp_synapse_params_1"]) # 入力層から興奮ニューロン
        self.obj["S_1_ei"] = Normal_Synapse(self.obj["N_1_exc"], self.obj["N_1_inh"], "exc", name="S_1_ei", connect="i==j", params=params["static_synapse_params_1ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_1_ie"] = Normal_Synapse(self.obj["N_1_inh"], self.obj["N_1_exc"], "inh", name="S_1_ie", connect="i!=j", params=params["static_synapse_params_1ie"]) # 側抑制
        self.obj["S_1_2"] = STDP_Synapse(self.obj["N_1_exc"], self.obj["N_2_exc"], name="S_1_2", connect="i==j", params=params["stdp_synapse_params_2"]) # １層から２層目の接続
        self.obj["S_2_ei"] = Normal_Synapse(self.obj["N_2_exc"], self.obj["N_2_inh"], "exc", name="S_2_ei", connect="i==j", params=params["static_synapse_params_2ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_2_ie"] = Normal_Synapse(self.obj["N_2_inh"], self.obj["N_2_exc"], "inh", name="S_2_ie", connect="i!=j", params=params["static_synapse_params_2ie"]) # 側抑制
        
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

class Center_Surround_WTA(Network_Frame):
    """
    抑制をCenter-Surroundで行うWTAネットワーク。
    """
    def __init__(self, enable_monitor:bool, params_json_path:str):
        # Make instances of neurons and synapses
        params = tools.load_parameters(params_json_path)
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_0"] = Poisson_Input(params["n_inp"], max_rate=params["max_rate"], name="N_0")
        self.obj["N_1"] = Conductance_LIF(params["n_e"], params["neuron_params_e"], name="N_1")
        self.obj["N_2"] = Conductance_LIF(params["n_i"], params["neuron_params_i"], name="N_2")

        self.obj["S_0"] = STDP_Synapse(self.obj["N_0"], self.obj["N_1"], name="S_0", connect=True, params=params["stdp_synapse_params"]) # 入力層から興奮ニューロン
        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", delay=0*ms, connect="i==j", params=params["static_synapse_params_ie"]) # 興奮ニューロンから抑制ニューロン
        k = params["static_synapse_params_ie"]["k"]
        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", delay=0*ms, connect=f"i!=j and abs(i-j)>{k}", params=params["static_synapse_params_ie"]) # Center-Surround抑制
        
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
        

class Mini_Columns_Network(Network_Frame):
    """
    複数のミニカラムから構成されるネットワーク
    """
    def __init__(self, enable_monitor:bool, params_json_path:str):
        pass

    class Mini_Column(Network_Frame):
        """
        一つのミニカラムのネットワーク
        WTA like ネットワーク
        """
        def __init__(self, enable_monitor:bool, params_json_path:str):
            self.enable_monitor = enable_monitor
            self.params = tools.load_parameters(params_json_path) # パラメータを読み込む
            
            self.obj = {} # ネットワークのオブジェクトを格納する辞書

            # Create network
            self.obj["N_1"] = Conductance_LIF(self.params["n_e"], self.params["neuron_params_e"], name="N_1")
            self.obj["N_2"] = Conductance_LIF(self.params["n_i"], self.params["neuron_params_i"], name="N_2")

            self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", connect="i==j", params=self.params["static_synapse_params_ei"]) # 興奮ニューロンから抑制ニューロン
            self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", connect="i!=j", params=self.params["static_synapse_params_ie"]) # 側抑制
            
            # Create monitors
            if enable_monitor:
                self.obj["spikemon_1"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_1")
                self.obj["spikemon_2"] = SpikeMonitor(self.obj["N_2"], record=True, name="spikemon_2")
                self.obj["statemon_1"] = StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name="statemon_1")
                self.obj["statemon_2"] = StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name="statemon_2")
            self.obj["spikemon_for_assign"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_for_assign") # ラベル割当に必要

            self.network = Network(self.obj.values()) # ネットワークを作成
            
        def set_input_neurons(self, neuron_group:NeuronGroup):
            """
            渡された入力ニューロンをこのミニカラムの入力層に接続して，
            入力層のニューロングループとシナプスをネットワークに追加します。
            
            Args:
                neuron_group (NeuronGroup): 入力層に接続するニューロングループ
            """
            self.obj["N_0"] = neuron_group
            self.obj["S_0"] = STDP_Synapse(self.obj["N_0"], self.obj["N_1"], name="S_0", connect="i==j", params=self.params["stdp_synapse_params"]) # 入力層から興奮ニューロン
            
            if self.enable_monitor: # モニタを追加
                self.obj["spikemon_0"] = SpikeMonitor(self.obj["N_0"], record=True, name="spikemon_0")
                self.obj["statemon_S"] = StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_S")
                self.network.add(self.obj["S_0"], self.obj["N_0"], self.obj["spikemon_0"], self.obj["statemon_S"])
            else:
                self.network.add(self.obj["S_0"], self.obj["N_0"])





