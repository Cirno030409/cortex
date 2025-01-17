import pprint

from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from brian2 import *
import Brian2_Framework.Tools as tools
from tqdm import tqdm

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
        self.network = Network()
        self.obj = {}
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
        
    def set_input_image(self, image:np.ndarray, spontaneous_rate:int=0):
        """
        入力画像を設定します。obj["N_inp"]が入力層であると想定しています。

        Args:
            image (np.ndarray): 入力画像\n
            spontaneous_rate (int, optional): 自発発火率. Defaults to 0.
        """
        self.obj["N_inp"].change_image(image, spontaneous_rate)
        
    def run(self, duration:int, *args, **kwargs):
        """
        ネットワークを実行します。

        Args:
            duration (int): 実行時間
        """

        self.network.run(duration, *args, **kwargs)
        
    def set_params(self, params: dict) -> None:
        """
        ネットワークのパラメータを設定します。

        Args:
            params (dict): キーをパラメータ名、値をパラメータ値とする辞書

        Raises:
            KeyError: 指定されたパラメータが存在しない場合
        """
        for key, value in params.items():
            try:
                setattr(self.network, key, value)
            except AttributeError:
                raise KeyError(f"パラメータ '{key}' が見つかりません。")
        
    def reset(self):
        """
        ネットワークをリセットします。
        """
        # ニューロン
        neurons = [obj for obj in self.network.objects if isinstance(obj, NeuronGroup)]

        # シナプス
        synapses = [obj for obj in self.network.objects if isinstance(obj, Synapses)]
        
        for i in range(len(neurons)):
            neurons[i].v = -60
            
        for i in range(len(synapses)):
            synapses[i].ge = 0
            synapses[i].gi = 0
            try:
                synapses[i].apre = 0
                synapses[i].apost = 0
            except:
                pass
            
class Mini_Column_biological_3inh(Network_Frame):
    """
    Jung H.Lee, Christof Koch, and Stefan MIhalas, 2017, "A Computational Analysis of the Functin of Three Inhibitory Cell Types in Contextual Visual Processing"
    のミニカラムネットワーク。
    """
    def __init__(self, params:dict):
        super().__init__()
        obj = {}
        
        # 層とニューロンタイプの定義
        layers = ["L23", "L4", "L5", "L6"]
        neuron_types = ["pv", "sst", "vip", "pyr", "inh"]
        
        # ニューロン作成
        # L2/3
        obj["L23_N_pv"] = Conductance_LIF_Neuron(params["L23"]["n_pv"]*params["network_scale"], params["neuron"], name="L23_N_pv")
        obj["L23_N_sst"] = Conductance_LIF_Neuron(params["L23"]["n_sst"]*params["network_scale"], params["neuron"], name="L23_N_sst")
        obj["L23_N_vip"] = Conductance_LIF_Neuron(params["L23"]["n_vip"]*params["network_scale"], params["neuron"], name="L23_N_vip")
        obj["L23_N_pyr"] = Conductance_LIF_Neuron(params["L23"]["n_pyr"]*params["network_scale"], params["neuron"], name="L23_N_pyr")
        
        # L4
        obj["L4_N_pyr"] = Conductance_LIF_Neuron(params["L4"]["n_pyr"]*params["network_scale"], params["neuron"], name="L4_N_pyr")
        obj["L4_N_inh"] = Conductance_LIF_Neuron(params["L4"]["n_inh"]*params["network_scale"], params["neuron"], name="L4_N_inh")
        
        # L5
        obj["L5_N_pyr"] = Conductance_LIF_Neuron(params["L5"]["n_pyr"]*params["network_scale"], params["neuron"], name="L5_N_pyr")
        obj["L5_N_inh"] = Conductance_LIF_Neuron(params["L5"]["n_inh"]*params["network_scale"], params["neuron"], name="L5_N_inh")
        
        # L6
        obj["L6_N_pyr"] = Conductance_LIF_Neuron(params["L6"]["n_pyr"]*params["network_scale"], params["neuron"], name="L6_N_pyr")
        obj["L6_N_inh"] = Conductance_LIF_Neuron(params["L6"]["n_inh"]*params["network_scale"], params["neuron"], name="L6_N_inh")
        
        # シナプス作成
        # 論文にあるシナプスの接続確率と重み使用してシナプスを作成
        for from_layer in tqdm(layers, desc="Constructing Network"):
            for from_neuron_type in neuron_types:
                for to_layer in layers:
                    for to_neuron_type in neuron_types:
                        # L2/3層以外にはPV, SST, VIPはない
                        if from_layer != "L23" and (from_neuron_type == "pv" or from_neuron_type == "vip" or from_neuron_type == "sst"):
                            continue
                        if to_layer != "L23" and (to_neuron_type == "pv" or to_neuron_type == "vip" or to_neuron_type == "sst"):
                            continue
                        if from_layer == "L23" and from_neuron_type == "inh":
                            continue
                        if to_layer == "L23" and to_neuron_type == "inh":
                            continue
                        # シナプス特性の設定 抑制性 or 興奮性 -> 抑制性 or 興奮性
                        if from_neuron_type == "pv" or from_neuron_type == "sst" or from_neuron_type == "vip" or from_neuron_type == "inh":
                            from_exc_or_inh = "inh"
                        elif from_neuron_type == "pyr":
                            from_exc_or_inh = "exc"
                        else:
                            raise ValueError("from_neuron_typeが不正です。")
                        if to_neuron_type == "pv" or to_neuron_type == "sst" or to_neuron_type == "vip" or to_neuron_type == "inh":
                            to_exc_or_inh = "inh"
                        elif to_neuron_type == "pyr":
                            to_exc_or_inh = "exc"
                        else:
                            raise ValueError("to_neuron_typeが不正です。")
                        # シナプスを作成
                        obj[f"{from_layer}_S_{from_neuron_type}_to_{to_layer}_{to_neuron_type}"] = Normal_Synapse(obj[f"{from_layer}_N_{from_neuron_type}"], 
                                                                                                                  obj[f"{to_layer}_N_{to_neuron_type}"], 
                                                                                                                  name=f"{from_layer}_S_{from_neuron_type}_to_{to_layer}_{to_neuron_type}", 
                                                                                                                  connect=True, 
                                                                                                                  p=params["synapse"]["conn_probability"][f"{from_exc_or_inh}->{to_exc_or_inh}"][from_layer][to_layer], # 接続確率
                                                                                                                  params={"w": params["synapse"]["weight"][f"{from_neuron_type}->{to_neuron_type}"]}, # 重み
                                                                                                                  exc_or_inh=from_exc_or_inh)
                        
        # モニター作成
        for key in params["monitor"].keys():
            self.network.add(
                SpikeMonitor(obj[key], record=params["monitor"][key], name=f"spikemon_{key}"),
            )
        print(obj.keys())
        self.network.add(obj.values())
        
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
    def __init__(self, enable_monitor:bool, params:dict):
        # Make instances of neurons and synapses
        super().__init__()
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_inp"] = Poisson_Input_Neuron(params["n_inp"], max_rate=params["max_rate"], name="N_inp")
        self.obj["N_1"] = Conductance_LIF_Neuron(params["n_e"], params["neuron_params_e"], name="N_1")
        self.obj["N_2"] = Conductance_LIF_Neuron(params["n_i"], params["neuron_params_i"], name="N_2")

        self.obj["S_0"] = STDP_Synapse(self.obj["N_inp"], self.obj["N_1"], name="S_0", connect=True, params=params["stdp_synapse_params"]) # 入力層から興奮ニューロン
        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], exc_or_inh="exc", name="S_1", delay=0*ms, connect="i==j", params=params["static_synapse_params_ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], exc_or_inh="inh", name="S_2", delay=0*ms, connect="i!=j", params=params["static_synapse_params_ie"]) # 側抑制
        
        # Create monitors
        if enable_monitor:
            self.network.add(
                SpikeMonitor(self.obj["N_inp"], record=True, name="spikemon_inp"),
                SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_N_1"),
                SpikeMonitor(self.obj["N_2"], record=True, name="spikemon_N_2"),
                StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name="statemon_N_1"),
                StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name="statemon_N_2"),
                StateMonitor(self.obj["S_0"], ["w", "apre", "apost"], record=0, name="statemon_S_0")
                )
        self.network.add(SpikeMonitor(self.obj["N_1"], record=True, name="spikemon_for_assign")) # ラベル割当に必要

        self.network.add(self.obj.values())
        
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
        self.obj["S_1_ei"] = Normal_Synapse(self.obj["N_1_exc"], self.obj["N_1_inh"], exc_or_inh="exc", name="S_1_ei", connect="i==j", params=params["static_synapse_params_1ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_1_ie"] = Normal_Synapse(self.obj["N_1_inh"], self.obj["N_1_exc"], exc_or_inh="inh", name="S_1_ie", connect="i!=j", params=params["static_synapse_params_1ie"]) # 側抑制
        self.obj["S_1_2"] = STDP_Synapse(self.obj["N_1_exc"], self.obj["N_2_exc"], name="S_1_2", connect="i==j", params=params["stdp_synapse_params_2"]) # １層から２層目の接続
        self.obj["S_2_ei"] = Normal_Synapse(self.obj["N_2_exc"], self.obj["N_2_inh"], exc_or_inh="exc", name="S_2_ei", connect="i==j", params=params["static_synapse_params_2ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_2_ie"] = Normal_Synapse(self.obj["N_2_inh"], self.obj["N_2_exc"], exc_or_inh="inh", name="S_2_ie", connect="i!=j", params=params["static_synapse_params_2ie"]) # 側抑制
        
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

    def __init__(self, enable_monitor:bool, params:dict, column_id:int):
        super().__init__()
        self.column_id = column_id
        self.enable_monitor = enable_monitor
        self.params = params # パラメータを読み込む
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書

        # Create network
        self.obj["N_1"] = Conductance_LIF_Neuron(self.params["n_e"], self.params["neuron_params_e"], name=f"mc{column_id}_N_1")
        self.obj["N_2"] = Conductance_LIF_Neuron(self.params["n_i"], self.params["neuron_params_i"], name=f"mc{column_id}_N_2")

        # 興奮性結合
        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], exc_or_inh="exc", name=f"mc{column_id}_S_1", connect="i==j", params=self.params["static_synapse_params_ei"]) # 興奮ニューロンから抑制ニューロン

        # 抑制性結合
        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], exc_or_inh="inh", name=f"mc{column_id}_S_2", connect=f"i!=j", params=self.params["static_synapse_params_ie"]) # 側抑制
        # self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], exc_or_inh="inh", name=f"mc{column_id}_S_2", connect=f"i-j <= {self.params['k']}", params=self.params["static_synapse_params_ie"]) # 隣を抑制
        # self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], exc_or_inh="inh", name=f"mc{column_id}_S_2", connect=f"i-j >= 25", params=self.params["static_synapse_params_ie"]) # Center-Surround抑制
        
        
        # Create monitors
        if self.enable_monitor:
            self.network.add(
                SpikeMonitor(self.obj["N_2"], record=True, name=f"mc{column_id}_spikemon_N_2"),
                StateMonitor(self.obj["N_1"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name=f"mc{column_id}_statemon_N_1"),
                StateMonitor(self.obj["N_2"], ["v",  "Ie", "Ii", "ge", "gi"], record=True, name=f"mc{column_id}_statemon_N_2")
            )
        self.network.add(SpikeMonitor(self.obj["N_1"], record=True, name=f"mc{column_id}_spikemon_for_assign")) # ラベル割当に必要

        self.network.add(self.obj.values())
        
    def get_network(self):
        return self.network
        
    def connect_neurons(self, source:NeuronGroup, connect_to:str, stdp_or_normal:str, syn_name:str, param_name:str, exc_or_inh:str="", connect:bool=True):
        """
        渡された入力ニューロンをこのミニカラムの指定されたオブジェクトに接続します。
        シナプスの種類と，結合方法を指定できます。JSONファイルに書かれたシナプスのパラメータ名を指定してください。
        シナプス名はmc{column_id}_{syn_name}となります。
        
        Args:
            source (NeuronGroup): 入力層に接続するニューロングループ
            connect_to (str): 接続するオブジェクトの名前
            stdp_or_normal (str): シナプスの種類。"stdp"か"normal"
            exc_or_inh (str): シナプスの興奮性か抑制性か。"exc"か"inh"
            connect (bool): シナプスを接続するかどうか
            param_name (str): シナプスのパラメータの名前
        """
        # 二つのニューロングループ間のシナプスを作成 & 接続
        if stdp_or_normal == "stdp":
            syn = STDP_Synapse(source, self.obj[connect_to], exc_or_inh=exc_or_inh, name=f"mc{self.column_id}_{syn_name}", connect=connect, params=self.params[param_name]) # 入力層から興奮ニューロン
            if self.enable_monitor:
                # self.network.add(StateMonitor(syn, ["w", "apre", "apost"], record=0, name=f"mc{self.column_id}_statemon_S"))
                pass
        else:
            syn = Normal_Synapse(source, self.obj[connect_to], exc_or_inh=exc_or_inh, name=f"mc{self.column_id}_{syn_name}", connect=connect, params=self.params[param_name]) # 入力層から興奮ニューロン
                    
        self.network.add(syn)

            
class Cortex(Network_Frame):
    """
    複数のミニカラムから構成されるネットワーク
    ミニカラムはMini_Columnクラスを使用しています。
    
    Args:
        enable_monitor (bool): モニタリングを有効にするか\n
        params_json_path (str): パラメータを保存したJSONファイルのパス
    """
    def __init__(self, enable_monitor:bool, params_cortex:dict, params_mc:dict):
        super().__init__()
        self.enable_monitor = enable_monitor
        self.params = params_cortex
        
        self.obj = {} # ネットワークのオブジェクトを格納する辞書
        self.columns = {} # ミニカラムのリスト
        
        self.obj["N_inp"] = Poisson_Input_Neuron(self.params["n_inp"], max_rate=self.params["max_rate"], name="N_inp")
        
        if self.enable_monitor:
            self.obj["spikemon_inp"] = SpikeMonitor(self.obj["N_inp"], record=True, name="spikemon_inp")
        
        # ミニカラムを作成して入力層と接続
        for i in range(self.params["n_mini_column"]):
            self.columns[i] = Mini_Column(self.enable_monitor, params_mc, column_id=i) # ミニカラムを作成
            self.columns[i].connect_neurons(self.obj["N_inp"], connect_to="N_1", stdp_or_normal="stdp", exc_or_inh="exc", syn_name="S_0", param_name="stdp_synapse_params", connect=True) # 接続
        
        
        # # ミニカラム間を結合
        for i in range(self.params["n_mini_column"]):
            for j in range(self.params["n_mini_column"]):
                if i != j:
                    self.columns[i].connect_neurons(source=self.columns[j].network[f"mc{j}_N_1"], connect_to="N_2", stdp_or_normal="normal", exc_or_inh="exc", syn_name=f"S_mc_{i}_{j}", param_name="inter_column_synapse_params_ei", connect=True)

        
        # ネットワークを作成
        self.network.add(self.obj.values())
        for column in self.columns.values():
            self.network.add(column.get_network())
            
class Optimized_Cortex(Network_Frame):
    #! 結局未実装。実装予定もなし。
    def __init__(self, enable_monitor:bool, params_cortex:dict, params_mc:dict):
        super().__init__()
        
        self.params_cortex = params_cortex
        self.params_mc = params_mc
        n_columns = params_cortex["n_mini_column"]
        n_exc_per_column = params_mc["n_e"]
        n_inh_per_column = params_mc["n_i"]
            
        # ニューロングループ
        self.obj["N_inp"] = Poisson_Input_Neuron(
            params_cortex["n_inp"], 
            max_rate=params_cortex["max_rate"], 
            name="N_inp"
        )
        self.obj["N_all_exc"] = Conductance_LIF_Neuron(
            n_columns * n_exc_per_column, 
            params_mc["neuron_params_e"], 
            name="N_all_exc"
        )
        self.obj["N_all_inh"] = Conductance_LIF_Neuron(
            n_columns * n_inh_per_column, 
            params_mc["neuron_params_i"], 
            name="N_all_inh"
        )
        
        # シナプス接続
        # 1. 入力層→興奮性ニューロン
        self.obj["S_inp_exc"] = STDP_Synapse(
            self.obj["N_inp"], 
            self.obj["N_all_exc"],
            name="S_inp_exc",
            connect=True,
            params=params_mc["stdp_synapse_params"]
        )
        
        # 2. 興奮性→抑制性（各ミニカラム内での1対1接続）
        connect_ei = '(i // n_exc_per_column) == (j // n_inh_per_column) and ((i % n_exc_per_column) == (j % n_inh_per_column))'
        
        self.obj["S_ei"] = Normal_Synapse(
            self.obj["N_all_exc"],
            self.obj["N_all_inh"],
            exc_or_inh="exc",
            name="S_ei",
            connect=connect_ei,
            params=params_mc["static_synapse_params_ei"],
            namespace={'n_exc_per_column': n_exc_per_column, 'n_inh_per_column': n_inh_per_column}
        )
        
        # 3. 抑制性→興奮性（各ミニカラム内でのWTA結合）
        connect_ie = '(i // n_inh_per_column) == (j // n_exc_per_column) and ((i % n_inh_per_column) != (j % n_exc_per_column))'
        
        self.obj["S_ie"] = Normal_Synapse(
            self.obj["N_all_inh"], 
            self.obj["N_all_exc"],
            exc_or_inh="inh",
            name="S_ie",
            connect=connect_ie,
            params=params_mc["static_synapse_params_ie"],
            namespace={'n_exc_per_column': n_exc_per_column, 'n_inh_per_column': n_inh_per_column}
        )
        
        # モニター
        self.obj["spikemon_inp"] = SpikeMonitor(self.obj["N_inp"], record=True, name="spikemon_inp")
        for i in range(n_columns):
            self.obj[f"mc{i}_spikemon_exc"] = SpikeMonitor(self.obj["N_all_exc"][self.get_column_neurons_slice(i)["exc"]], record=True, name=f"mc{i}_spikemon_exc")
            if enable_monitor:
                self.obj[f"mc{i}_spikemon_inh"] = SpikeMonitor(self.obj["N_all_inh"][self.get_column_neurons_slice(i)["inh"]], record=True, name=f"mc{i}_spikemon_inh")
        
        self.network.add(self.obj.values())
        
    def get_network(self):
        return self.network
    
    def get_column_neurons_slice(self, column_id: int) -> dict:
        """
        指定したカラムのニューロンを抽出します。
        スライスオブジェクトを返します。

        Args:
            column_id (int): カラムのID

        Returns:
            dict: {"exc": 興奮性ニューロンのスライス, "inh": 抑制性ニューロンのスライス}
        """
        # ニューロンの数を取得
        n_exc_per_column = self.obj["N_all_exc"].N // self.params_cortex["n_mini_column"]
        n_inh_per_column = self.obj["N_all_inh"].N // self.params_cortex["n_mini_column"]
        
        # ニューロンのインデックスを取得
        exc_start = column_id * n_exc_per_column
        exc_end = (column_id + 1) * n_exc_per_column
        inh_start = column_id * n_inh_per_column
        inh_end = (column_id + 1) * n_inh_per_column
        
        return {
            "exc": slice(exc_start, exc_end),
            "inh": slice(inh_start, inh_end)
        }
        
    def get_synaptic_weight(self, name: str, n_column:int, type:str) -> np.ndarray:
        """
        指定したシナプスの重みを取得します。
        """
        return self.network[name].w[self.get_column_synapse_slice(n_column)[type]]
    
    def get_column_synapses(self, column_id: int, name: str) -> Synapses:
        """
        指定したカラムに属するシナプスのみを抽出して新しいSynapsesオブジェクトを作成します。

        Args:
            column_idx (int): カラムのインデックス
            name (str): シナプスの名前

        Returns:
            Synapses: 抽出された新しいシナプスオブジェクト
        """
        source_synapse = self.network[name]
        # カラム内のニューロンのインデックス範囲を取得
        neurons = self.get_column_neurons_slice(column_id)
        
        # シナプスの接続インデックスを取得
        i, j = source_synapse.i[:], source_synapse.j[:]
        w = source_synapse.w[:]
        
        # カラム内のシナプスのみを抽出するマスクを作成
        mask = (i >= neurons["exc"].start) & (i < neurons["exc"].stop) & \
            (j >= neurons["inh"].start) & (j < neurons["inh"].stop)
            

        print("\n",  neurons["exc"].start, neurons["exc"].stop, neurons["inh"].start, neurons["inh"].stop, "\n")
        
        # 新しいシナプスオブジェクトを作成
        new_synapse = Synapses(
            source_synapse.source,
            source_synapse.target,
            model=source_synapse.model,
            on_pre=source_synapse.on_pre,
            on_post=source_synapse.on_post,
            name=f"mc{column_id}_{source_synapse.name}"
        )
        
        # 抽出したシナプスの接続を設定
        new_synapse.connect(i=i[mask], j=j[mask])
        new_synapse.w = w[mask]
        
        return new_synapse
        
    def get_column_synapse_slice(self, column_id: int) -> dict:
        """
        指定したカラムのシナプスのインデックスを取得します。
        スライスオブジェクトを返します。

        Args:
            column_id (int): カラムのID

        Returns:
            dict: {"inp_exc": 入力→興奮性のシナプスのスライス, 
                  "ei": 興奮性→抑制性のシナプスのスライス, 
                  "ie": 抑制性→興奮性のシナプスのスライス}
        """
        neurons = self.get_column_neurons_slice(column_id)
        
        synapse_slice = {
            "inp_exc": slice(neurons["exc"].start * self.params_cortex["n_inp"], 
                           neurons["exc"].stop * self.params_cortex["n_inp"]),
            "ei": slice(neurons["exc"].start * self.params_mc["n_i"], 
                       neurons["exc"].stop * self.params_mc["n_i"]),
            "ie": slice(neurons["inh"].start * self.params_mc["n_e"], 
                       neurons["inh"].stop * self.params_mc["n_e"])
        }
        
        return synapse_slice

    def get_column_weights(self, column_id: int) -> dict:
        """
        指定したカラムのシナプス重みを取得します。

        Args:
            column_id (int): カラムのID

        Returns:
            dict: {"input_exc": 入力→興奮性の重み, "exc_inh": 興奮性→抑制性の重み, "inh_exc": 抑制性→興奮性の重み}
        """
                
        weights = {
            "input_exc": self.obj["S_inp_exc"].w[self.get_column_synapse_slice(column_id)["inp_exc"]],
            "exc_inh": self.obj["S_ei"].w[self.get_column_synapse_slice(column_id)["ei"]],
            "inh_exc": self.obj["S_ie"].w[self.get_column_synapse_slice(column_id)["ie"]]
        }
        
        return weights
        





