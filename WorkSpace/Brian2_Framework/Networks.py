import pprint

from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from brian2 import *
import Brian2_Framework.Tools as tools

class Network_Frame(Network):
    """
    ネットワークに一律の機能を提供する親クラス
    
    入力ニューロンのオブジェクト名は"N_inp"
    学習するSTDPシナプスのオブジェクト名は"S_0"であると想定しています。
    
    Methods:
        enable_learning(): 学習を有効にします。\n
        disable_learning(): 学習を無効にします。\n
        change_image(image:np.ndarray, spontaneous_rate:int=0): 入力画像を変更します。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
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
        入力画像を変更します。obj["N_inp"]が入力層であると想定しています。

        Args:
            image (np.ndarray): 入力画像\n
            spontaneous_rate (int, optional): 自発発火率. Defaults to 0.
        """
        self.obj["N_inp"].change_image(image, spontaneous_rate)
        
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
        self.obj["N_inp"] = Poisson_Input_Neuron(params["n_inp"], max_rate=params["max_rate"], name="N_inp")
        self.obj["N_1"] = Conductance_LIF_Neuron(params["n_e"], params["neuron_params_e"], name="N_1")
        self.obj["N_2"] = Conductance_LIF_Neuron(params["n_i"], params["neuron_params_i"], name="N_2")

        self.obj["S_0"] = STDP_Synapse(self.obj["N_inp"], self.obj["N_1"], name="S_0", connect=True, params=params["stdp_synapse_params"]) # 入力層から興奮ニューロン
        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", delay=0*ms, connect="i==j", params=params["static_synapse_params_ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", delay=0*ms, connect="i!=j", params=params["static_synapse_params_ie"]) # 側抑制
        
        # Create monitors
        if enable_monitor:
            self.network.add(
                SpikeMonitor(self.obj["N_inp"], record=True, name="spikemon_input"),
                SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_1"),
                SpikeMonitor(self.obj["N_2"], record=True, name="spikemon_2"),
                StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_1"),
                StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_2"),
                StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_S")
                )
        self.network.add(SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_for_assign")) # ラベル割当に必要

        self.network = Network(self.obj.values()) # ネットワークを作成
        
class Chunk_WTA(Network_Frame):
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
        self.obj["N_inp"] = self.neuron_inp(params["n_inp"], max_rate=params["max_rate"], name="N_inp")
        self.obj["N_1_exc"] = Conductance_LIF_Neuron(params["n_e"], params["neuron_params_1e"], name="N_1_exc")
        self.obj["N_1_inh"] = Conductance_LIF_Neuron(params["n_i"], params["neuron_params_1i"], name="N_1_inh")
        self.obj["N_2_exc"] = Conductance_LIF_Neuron(params["n_e"], params["neuron_params_2e"], name="N_2_exc")
        self.obj["N_2_inh"] = Conductance_LIF_Neuron(params["n_i"], params["neuron_params_2i"], name="N_2_inh")

        ## Synapses
        self.obj["S_0"] = STDP_Synapse(self.obj["N_inp"], self.obj["N_1_exc"], name="S_0", connect=True, params=params["stdp_synapse_params_1"]) # 入力層から興奮ニューロン
        self.obj["S_1_ei"] = Normal_Synapse(self.obj["N_1_exc"], self.obj["N_1_inh"], "exc", name="S_1_ei", connect="i==j", params=params["static_synapse_params_1ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_1_ie"] = Normal_Synapse(self.obj["N_1_inh"], self.obj["N_1_exc"], "inh", name="S_1_ie", connect="i!=j", params=params["static_synapse_params_1ie"]) # 側抑制
        self.obj["S_1_2"] = STDP_Synapse(self.obj["N_1_exc"], self.obj["N_2_exc"], name="S_1_2", connect="i==j", params=params["stdp_synapse_params_2"]) # １層から２層目の接続
        self.obj["S_2_ei"] = Normal_Synapse(self.obj["N_2_exc"], self.obj["N_2_inh"], "exc", name="S_2_ei", connect="i==j", params=params["static_synapse_params_2ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_2_ie"] = Normal_Synapse(self.obj["N_2_inh"], self.obj["N_2_exc"], "inh", name="S_2_ie", connect="i!=j", params=params["static_synapse_params_2ie"]) # 側抑制
        
        # Create monitors
        if enable_monitor:
            self.obj["spikemon_inp"] = SpikeMonitor(self.obj["N_inp"], record=True, name="spikemon_inp")
            self.obj["spikemon_1_exc"] = SpikeMonitor(self.obj["N_1_exc"], record=True, name="spikemon_1_exc")
            self.obj["statemon_1stdp"] = StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_1stdp")
            self.obj["statemon_2stdp"] = StateMonitor(self.obj["S_1_2"], ["w", "apre", "apost"], record=0, name="statemon_2stdp")
            self.obj["statemon_N_1_exc"] = StateMonitor(self.obj["N_1_exc"], ["v", "Ie", "Ii", "ge", "gi"], record=50, name="statemon_N_1_exc")
        self.obj["spikemon_2_exc"] = SpikeMonitor(self.obj["N_2_exc"], record=True, name="spikemon_2_exc") # ラベル割当に必要

        self.network = Network(self.obj.values()) # ネットワークを作成


class Center_Surround_WTA(Network_Frame):
    """
    抑制をCenter-Surroundで行うWTAネットワーク。
    
    Args:
        enable_monitor (bool): モニタリングを有効にするか\n
        params_json_path (str): パラメータを保存したJSONファイルのパス
    """
    def __init__(self, enable_monitor:bool, params_json_path:str):
        # Make instances of neurons and synapses
        params = tools.load_parameters(params_json_path)
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_inp"] = Poisson_Input_Neuron(params["n_inp"], max_rate=params["max_rate"], name="N_inp")
        self.obj["N_1"] = Conductance_LIF_Neuron(params["n_e"], params["neuron_params_e"], name="N_1")
        self.obj["N_2"] = Conductance_LIF_Neuron(params["n_i"], params["neuron_params_i"], name="N_2")

        self.obj["S_0"] = STDP_Synapse(self.obj["N_inp"], self.obj["N_1"], name="S_0", connect=True, params=params["stdp_synapse_params"]) # 入力層から興奮ニューロン
        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], "exc", name="S_1", delay=0*ms, connect="i==j", params=params["static_synapse_params_ie"]) # 興奮ニューロンから抑制ニューロン
        k = params["static_synapse_params_ie"]["k"]
        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], "inh", name="S_2", delay=0*ms, connect=f"i!=j and abs(i-j)>{k}", params=params["static_synapse_params_ie"]) # Center-Surround抑制
        
        # Create monitors
        if enable_monitor:
            self.obj["spikemon_inp"] = SpikeMonitor(self.obj["N_inp"], record=True, name="spikemon_inp")
            self.obj["spikemon_1"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_1")
            self.obj["spikemon_2"] = SpikeMonitor(self.obj["N_2"], record=True, name="spikemon_2")
            self.obj["statemon_1"] = StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_1")
            self.obj["statemon_2"] = StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=50, name="statemon_2")
            self.obj["statemon_S"] = StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_S")
        self.obj["spikemon_for_assign"] = SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_for_assign") # ラベル割当に必要

        self.network = Network(self.obj.values()) # ネットワークを作成
        
        
class Mini_Column(Network_Frame):
    """
    一つのミニカラムのネットワーク
    WTA like ネットワーク
    
    Args:
        enable_monitor (bool): モニタリングを有効にするか\n
        params_json_path (str): パラメータを保存したJSONファイルのパス
        
    Methods:
        connect_neurons(neuron_group:NeuronGroup, object_name:str, stdp_or_normal:str, param_name:str, connect:bool=True): ニューロングループを接続します。
    """

    def __init__(self, enable_monitor:bool, params_json_path:str, column_id:int):
        self.column_id = column_id
        self.enable_monitor = enable_monitor
        self.network = Network()
        self.params = tools.load_parameters(params_json_path) # パラメータを読み込む
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_1"] = Conductance_LIF_Neuron(self.params["n_e"], self.params["neuron_params_e"], name=f"mc{column_id}_N_1")
        self.obj["N_2"] = Conductance_LIF_Neuron(self.params["n_i"], self.params["neuron_params_i"], name=f"mc{column_id}_N_2")

        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], "exc", name=f"mc{column_id}_S_1", connect="i==j", params=self.params["static_synapse_params_ei"]) # 興奮ニューロンから抑制ニューロン

        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], "inh", name=f"mc{column_id}_S_2", connect="i!=j", params=self.params["static_synapse_params_ie"]) # 側抑制
        
        # Create monitors
        if self.enable_monitor:
            self.network.add(SpikeMonitor(self.obj["N_1"], record=True, name=f"mc{column_id}_spikemon_1"),
                             SpikeMonitor(self.obj["N_2"], record=True, name=f"mc{column_id}_spikemon_2"),
                             StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name=f"mc{column_id}_statemon_1"),
                             StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name=f"mc{column_id}_statemon_2"))
        self.network.add(SpikeMonitor(self.obj["N_1"], record=True, name=f"mc{column_id}_spikemon_for_assign")) # ラベル割当に必要

        self.network = Network(self.obj.values()) # ネットワークを作成
        
    def get_network(self):
        return self.network
        
    def connect_neurons(self, neuron_group:NeuronGroup, connect_to:str, stdp_or_normal:str, param_name:str, connect:bool=True):
        """
        渡された入力ニューロンをこのミニカラムの指定されたオブジェクトに接続します。
        シナプスの種類と，結合方法を指定できます。JSONファイルに書かれたシナプスのパラメータ名を指定してください。
        
        Args:
            neuron_group (NeuronGroup): 入力層に接続するニューロングループ
            connect_to (str): 接続するオブジェクトの名前
            stdp_or_normal (str): シナプスの種類。"stdp"か"normal"
            connect (bool): シナプスを接続するかどうか
            param_name (str): シナプスのパラメータの名前
        """
        if stdp_or_normal == "stdp":
            self.network.add(STDP_Synapse(neuron_group, self.obj["N_1"], name=f"mc{self.column_id}_S_0", connect=connect, params=self.params[param_name])) # 入力層から興奮ニューロン
        else:
            self.network.add(Normal_Synapse(neuron_group, self.obj["N_1"], name=f"mc{self.column_id}_S_0", connect=connect, params=self.params[param_name])) # 入力層から興奮ニューロン
        
        if self.enable_monitor: # モニタを追加
            self.network.add(SpikeMonitor(neuron_group, record=True, name=f"mc{self.column_id}_spikemon_N_inp"))
            self.network.add(SpikeMonitor(self.obj["N_1"], record=True, name=f"mc{self.column_id}_spikemon_N_1"))
        else:
            self.network.add(self.obj["S_0"], neuron_group)
            
class Cortex(Network_Frame):
    """
    複数のミニカラムから構成されるネットワーク
    ミニカラムはMini_Columnクラスを使用しています。
    
    Args:
        enable_monitor (bool): モニタリングを有効にするか\n
        params_json_path (str): パラメータを保存したJSONファイルのパス
    """
    def __init__(self, enable_monitor:bool, params_json_path:str):
        super().__init__()
        self.enable_monitor = enable_monitor
        self.params = tools.load_parameters(params_json_path)
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書
        self.columns = {} # ミニカラムのリスト
        
        self.obj["N_inp"] = Poisson_Input_Neuron(self.params["n_inp"], max_rate=self.params["max_rate"], name="N_inp")
        
        for i in range(self.params["n_mini_column"]): # ミニカラムを作成
            mc = Mini_Column(self.enable_monitor, self.params["mini_column_params_json_path"], column_id=i) # ミニカラムを作成
            mc.connect_neurons(self.obj["N_inp"], connect_to="N_1", stdp_or_normal="stdp", param_name="stdp_synapse_params", connect=True) # 接続
            self.columns[i] = mc.get_network() # ネットワークを取得
        self.network = Network(self.obj.values(), self.columns.values()) # ネットワークを作成
        





