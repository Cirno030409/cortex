import pprint

from brian2 import *
from tqdm import tqdm

from Brian2_Framework.Neurons import *
from Brian2_Framework.Synapses import *
from Brian2_Framework.Tools import *


class Network_Frame(Network):
    """
    ネットワークに一律の機能を提供する親クラス
    
    入力ニューロンのオブジェクト名は"N_inp"
    学習するSTDPシナプスのオブジェクト名は"S_0"であると想定しています。
    
    Methods:
        enable_learning(): 学習を有効にします。\n
        disable_learning(): 学習を無効にします。\n
        change_image(image:np.ndarray, spontaneous_rate:int=0): 入力画像を変更します。
        set_input_neuron_rate(rate:float): 入力ニューロンの発火率を設定します。
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
        
    def set_input_neuron_rate(self, rate:float):
        """
        入力ニューロンの発火率を設定します。
        """
        self.obj["N_inp"].set_rate(rate)
        
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
            neurons[i].v = -60*mV
            
        for i in range(len(synapses)):
            synapses[i].ge = 0*nS
            synapses[i].gi = 0*nS
            try:
                synapses[i].apre = 0*nS
                synapses[i].apost = 0*nS
            except:
                pass
            
class Jung_H_Lee_Cortex_MicroCircuit_multiple(Network_Frame):
    """
    複数の新皮質の局所回路ネットワークを接続したネットワーク。
    """
    def __init__(self, params:dict):
        super().__init__()
        params_mc = load_parameters(params["micro_circuit_params_path"])
        
        for i in range(params["n_micro_circuit"]):
            self.network.add(Jung_H_Lee_Cortex_MicroCircuit(params_mc, circuit_id=i).network)
            
        # 局所回路間シナプス接続
        # コネクションの書式：[(from_neuron_type, to_neuron_type, micro_circuit_id_diff_connect_to(or "all")]
        connections = [("pyr", "pyr", 1), ("pyr", "sst", "all"), ("pyr", "pv", 1), ("pyr", "pv", 2), ("pv", "pyr", 1)]
        
        # すでに接続されているシナプスを追跡するための集合
        connected_synapses = set()
        
        for connection in tqdm(connections, desc="Connecting Micro Circuits"):
            from_neuron_type, to_neuron_type, circuit_id_diff = connection
            for i in range(params["n_micro_circuit"]):
                if circuit_id_diff == "all":
                    connect_to = [j for j in range(params["n_micro_circuit"]) if j != i] # 全ての局所回路と接続
                else:
                    if (i + circuit_id_diff) % params["n_micro_circuit"] == (i - circuit_id_diff) % params["n_micro_circuit"]:
                        connect_to = [(i + circuit_id_diff) % params["n_micro_circuit"]]
                    else:
                        connect_to = [(i + circuit_id_diff) % params["n_micro_circuit"]] + [(i - circuit_id_diff) % params["n_micro_circuit"]]
                for j in connect_to:
                    if i == j:
                        continue
                    
                    # 接続の一意の識別子を作成
                    connection_id = f"M{i}_{from_neuron_type}_to_M{j}_{to_neuron_type}"
                    
                    # すでに接続が存在する場合はスキップ
                    if connection_id in connected_synapses:
                        continue
                    
                    # 接続を追跡集合に追加
                    connected_synapses.add(connection_id)
                    
                    if from_neuron_type == "pyr":
                        self.network.add(Normal_Synapse(self.network[f"M{i}_L23_N_{from_neuron_type}"], self.network[f"M{j}_L23_N_{to_neuron_type}"], name=f"M{i}_S_{from_neuron_type}_to_M{j}_S_{to_neuron_type}", connect=True, p=1, params={"w_ave": params_mc["synapse"]["weight"]["default_exc"]["ave"], "w_std": params_mc["synapse"]["weight"]["default_exc"]["std"], "delay_ave": params_mc["synapse"]["delay"]["intra_column_exc"]["ave"], "delay_std": params_mc["synapse"]["delay"]["intra_column_exc"]["std"], "tau": params_mc["synapse"]["decay_time"]["default_exc"]}, exc_or_inh="exc"))
                    else:
                        self.network.add(Normal_Synapse(self.network[f"M{i}_L23_N_{from_neuron_type}"], self.network[f"M{j}_L23_N_{to_neuron_type}"], name=f"M{i}_S_{from_neuron_type}_to_M{j}_S_{to_neuron_type}", connect=True, p=1, params={"w_ave": params_mc["synapse"]["weight"]["default_inh"]["ave"], "w_std": params_mc["synapse"]["weight"]["default_inh"]["std"], "delay_ave": params_mc["synapse"]["delay"]["intra_column_inh"]["ave"], "delay_std": params_mc["synapse"]["delay"]["intra_column_inh"]["std"], "tau": params_mc["synapse"]["decay_time"]["default_inh"]}, exc_or_inh="inh"))
class Jung_H_Lee_Cortex_MicroCircuit(Network_Frame):
    """
    Jung H.Lee, Christof Koch, and Stefan MIhalas, 2017, "A Computational Analysis of the Function of Three Inhibitory Cell Types in Contextual Visual Processing"
    のにある新皮質の局所回路ネットワーク。
    複数の局所回路を同じネットワークとして接続する場合，一意のcircuit_idを指定する必要があります。
    
    Parameters:
        params (dict): パラメータ
        circuit_id (int, optional): 局所回路の一意のID. Defaults to 0.
        
    Methods:
        set_input_neuron_rate(rate:float): 入力ニューロンの発火率を設定します。
    """
    def __init__(self, params:dict, circuit_id:int=0):
        super().__init__()
        self.params = params
        self.circuit_id = circuit_id
        obj = {}
        # 層とニューロンタイプの定義
        layers = ["L23", "L4", "L5", "L6"]
        neuron_types = ["exc", "inh"]
        l23_neuron_types = ["pyr", "pv", "sst", "vip"]
        
        # ニューロン作成
        ## L2/3
        obj["L23_N_pv"] = Current_LIF_Neuron(int(params["L23"]["n_pv"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L23_N_pv")
        obj["L23_N_sst"] = Current_LIF_Neuron(int(params["L23"]["n_sst"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L23_N_sst")
        obj["L23_N_vip"] = Current_LIF_Neuron(int(params["L23"]["n_vip"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L23_N_vip")
        obj["L23_N_pyr"] = Current_LIF_Neuron(int(params["L23"]["n_pyr"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L23_N_pyr")
        
        ## L4
        obj["L4_N_exc"] = Current_LIF_Neuron(int(params["L4"]["n_exc"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L4_N_exc")
        obj["L4_N_inh"] = Current_LIF_Neuron(int(params["L4"]["n_inh"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L4_N_inh")
        
        ## L5
        obj["L5_N_exc"] = Current_LIF_Neuron(int(params["L5"]["n_exc"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L5_N_exc")
        obj["L5_N_inh"] = Current_LIF_Neuron(int(params["L5"]["n_inh"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L5_N_inh")

        ## L6
        obj["L6_N_exc"] = Current_LIF_Neuron(int(params["L6"]["n_exc"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L6_N_exc")
        obj["L6_N_inh"] = Current_LIF_Neuron(int(params["L6"]["n_inh"]*params["network_scale"]), params["neuron"], name=f"M{circuit_id}_L6_N_inh")
        
        ## Input Neuron
        obj["N_inp"] = Poisson_Input_Neuron(int(params["N_inp"]*params["network_scale"]), name=f"M{circuit_id}_N_inp")

        ## External Input Neuron for spontaneous activity
        ### for L2/3
        obj["L23_N_noise_to_pyr"] = Poisson_Input_Neuron(int(params["ex_fibers"]["num"]["L23"]["pyr"]*params["network_scale"]), name=f"M{circuit_id}_L23_N_noise_to_pyr")
        obj["L23_N_noise_to_pyr"].set_rate(params["ex_fibers"]["rate"]["L23"]["pyr"])
        obj["L23_N_noise_to_pv"] = Poisson_Input_Neuron(int(params["ex_fibers"]["num"]["L23"]["inh"]*params["network_scale"]), name=f"M{circuit_id}_L23_N_noise_to_pv")
        obj["L23_N_noise_to_pv"].set_rate(params["ex_fibers"]["rate"]["L23"]["pv"])
        obj["L23_N_noise_to_sst"] = Poisson_Input_Neuron(int(params["ex_fibers"]["num"]["L23"]["inh"]*params["network_scale"]), name=f"M{circuit_id}_L23_N_noise_to_sst")
        obj["L23_N_noise_to_sst"].set_rate(params["ex_fibers"]["rate"]["L23"]["sst"])
        obj["L23_N_noise_to_vip"] = Poisson_Input_Neuron(int(params["ex_fibers"]["num"]["L23"]["inh"]*params["network_scale"]), name=f"M{circuit_id}_L23_N_noise_to_vip")
        obj["L23_N_noise_to_vip"].set_rate(params["ex_fibers"]["rate"]["L23"]["vip"])
        ### for Other layer
        for layer in ["L4", "L5", "L6"]:
            obj[f"{layer}_N_noise_to_exc"] = Poisson_Input_Neuron(int(params["ex_fibers"]["num"][layer]["exc"]*params["network_scale"]), name=f"M{circuit_id}_{layer}_N_noise_to_exc")
            obj[f"{layer}_N_noise_to_exc"].set_rate(params["ex_fibers"]["rate"]["other"]["exc"])
            obj[f"{layer}_N_noise_to_inh"] = Poisson_Input_Neuron(int(params["ex_fibers"]["num"][layer]["inh"]*params["network_scale"]), name=f"M{circuit_id}_{layer}_N_noise_to_inh")
            obj[f"{layer}_N_noise_to_inh"].set_rate(params["ex_fibers"]["rate"]["other"]["inh"])
        
        # シナプス作成
        ## 外部ファイバー接続
        ### L2/3
        w_ave, w_std = self.load_synaptic_weight("ex_fibers", from_exc_or_inh="exc")
        delay_ave, delay_std = self.load_synaptic_delay("exc")
        obj["L23_S_noise_to_pyr"] = Normal_Synapse(obj["L23_N_noise_to_pyr"], obj["L23_N_pyr"], name=f"M{circuit_id}_L23_S_noise_to_pyr", connect=True, p=1, params={"w_ave": w_ave, "w_std": w_std, "delay_ave": delay_ave, "delay_std": delay_std, "tau": params["synapse"]["decay_time"]["default_exc"]}, exc_or_inh="exc")
        obj["L23_S_noise_to_pv"] = Normal_Synapse(obj["L23_N_noise_to_pv"], obj["L23_N_pv"], name=f"M{circuit_id}_L23_S_noise_to_pv", connect=True, p=1, params={"w_ave": w_ave, "w_std": w_std, "delay_ave": delay_ave, "delay_std": delay_std, "tau": params["synapse"]["decay_time"]["default_exc"]}, exc_or_inh="exc")
        obj["L23_S_noise_to_sst"] = Normal_Synapse(obj["L23_N_noise_to_sst"], obj["L23_N_sst"], name=f"M{circuit_id}_L23_S_noise_to_sst", connect=True, p=1, params={"w_ave": w_ave, "w_std": w_std, "delay_ave": delay_ave, "delay_std": delay_std, "tau": params["synapse"]["decay_time"]["default_exc"]}, exc_or_inh="exc")
        obj["L23_S_noise_to_vip"] = Normal_Synapse(obj["L23_N_noise_to_vip"], obj["L23_N_vip"], name=f"M{circuit_id}_L23_S_noise_to_vip", connect=True, p=1, params={"w_ave": w_ave, "w_std": w_std, "delay_ave": delay_ave, "delay_std": delay_std, "tau": params["synapse"]["decay_time"]["default_exc"]}, exc_or_inh="exc")
        ### Other Layer
        for layer in ["L4", "L5", "L6"]:
            try:
                w_ave, w_std = self.load_synaptic_weight("ex_fibers", from_exc_or_inh="exc")
                delay_ave, delay_std = self.load_synaptic_delay("exc")
                obj[f"{layer}_S_noise_to_exc"] = Normal_Synapse(obj[f"{layer}_N_noise_to_exc"], obj[f"{layer}_N_exc"], 
                                                                name=f"M{circuit_id}_{layer}_S_noise_to_exc", 
                                                                p=1, 
                                                                connect=True, 
                                                                params={"w_ave": w_ave, 
                                                                       "w_std": w_std,
                                                                       "delay_ave": delay_ave,
                                                                       "delay_std": delay_std,
                                                                       "tau": params["synapse"]["decay_time"]["default_exc"]}, 
                                                                exc_or_inh="exc")
                w_ave, w_std = self.load_synaptic_weight("ex_fibers", from_exc_or_inh="inh")
                delay_ave, delay_std = self.load_synaptic_delay("inh")
                obj[f"{layer}_S_noise_to_inh"] = Normal_Synapse(obj[f"{layer}_N_noise_to_inh"], 
                                                                obj[f"{layer}_N_inh"], 
                                                                name=f"M{circuit_id}_{layer}_S_noise_to_inh", 
                                                                p=1, 
                                                                connect=True, 
                                                                params={"w_ave": w_ave, 
                                                                       "w_std": w_std,
                                                                       "delay_ave": delay_ave,
                                                                       "delay_std": delay_std,
                                                                       "tau": params["synapse"]["decay_time"]["default_exc"]}, 
                                                                exc_or_inh="exc")
            except KeyError:
                print(f"シナプス接続が存在しません: {layer}_S_noise_to_exc")
                continue

        # Input neuron 接続
        obj["S_inp_to_L4_exc"] = Normal_Synapse(obj["N_inp"], obj["L4_N_exc"], name=f"M{circuit_id}_S_inp_to_L4_exc", connect=True, p=params["synapse"]["conn_probability"]["inp->exc"]["L4"], params={"w_ave": params["synapse"]["weight"]["default_exc"]["ave"], "w_std": params["synapse"]["weight"]["default_exc"]["std"], "tau": params["synapse"]["decay_time"]["default_exc"], "delay_ave": params["synapse"]["delay"]["intra_column_exc"]["ave"], "delay_std": params["synapse"]["delay"]["intra_column_exc"]["std"]}, exc_or_inh="exc")
        obj["S_inp_to_L6_exc"] = Normal_Synapse(obj["N_inp"], obj["L6_N_exc"], name=f"M{circuit_id}_S_inp_to_L6_exc", connect=True, p=params["synapse"]["conn_probability"]["inp->exc"]["L6"], params={"w_ave": params["synapse"]["weight"]["default_exc"]["ave"], "w_std": params["synapse"]["weight"]["default_exc"]["std"], "tau": params["synapse"]["decay_time"]["default_exc"], "delay_ave": params["synapse"]["delay"]["intra_column_exc"]["ave"], "delay_std": params["synapse"]["delay"]["intra_column_exc"]["std"]}, exc_or_inh="exc")
        obj["S_inp_to_L4_inh"] = Normal_Synapse(obj["N_inp"], obj["L4_N_inh"], name=f"M{circuit_id}_S_inp_to_L4_inh", connect=True, p=params["synapse"]["conn_probability"]["inp->inh"]["L4"], params={"w_ave": params["synapse"]["weight"]["default_exc"]["ave"], "w_std": params["synapse"]["weight"]["default_exc"]["std"], "tau": params["synapse"]["decay_time"]["default_exc"], "delay_ave": params["synapse"]["delay"]["intra_column_exc"]["ave"], "delay_std": params["synapse"]["delay"]["intra_column_exc"]["std"]}, exc_or_inh="exc")
        obj["S_inp_to_L6_inh"] = Normal_Synapse(obj["N_inp"], obj["L6_N_inh"], name=f"M{circuit_id}_S_inp_to_L6_inh", connect=True, p=params["synapse"]["conn_probability"]["inp->inh"]["L6"], params={"w_ave": params["synapse"]["weight"]["default_exc"]["ave"], "w_std": params["synapse"]["weight"]["default_exc"]["std"], "tau": params["synapse"]["decay_time"]["default_exc"], "delay_ave": params["synapse"]["delay"]["intra_column_exc"]["ave"], "delay_std": params["synapse"]["delay"]["intra_column_exc"]["std"]}, exc_or_inh="exc")
        
        ## 論文にあるシナプスの接続確率と重みを使用してシナプスを作成
        for from_layer in tqdm(layers, desc="Constructing Network"):
            for from_neuron_type in neuron_types:
                for to_layer in layers:
                    for to_neuron_type in neuron_types:
                        # シナプス接続をしない組み合わせをチェック
                        ## L2/3層への"exc"ニューロンをターゲットにした接続はしない
                        if (from_layer == "L23" and from_neuron_type == "exc") or (to_layer == "L23" and to_neuron_type == "exc"):
                            continue
                        ## ほかの層への"pyr"ニューロンをターゲットにした接続はしない
                        if (from_layer != "L23" and from_neuron_type == "pyr") or (to_layer != "L23" and to_neuron_type == "pyr"):
                            continue
                        # シナプス種の設定
                        ## pv, sst, vip, inhからのシナプスは抑制にする
                        if from_neuron_type == "inh":
                            from_exc_or_inh = "inh"
                        ## pyrからのシナプスは興奮にする
                        elif from_neuron_type == "exc":
                            from_exc_or_inh = "exc"
                        else:
                            raise ValueError("from_neuron_typeが不正です。")
                        if to_neuron_type == "inh":
                            to_exc_or_inh = "inh"
                        elif to_neuron_type == "pyr" or to_neuron_type == "exc":
                            to_exc_or_inh = "exc"
                        else:
                            raise ValueError("to_neuron_typeが不正です。")
                        # シナプスを接続
                        try: # 存在しないニューロングループは接続しない（デバッグ用）
                            if from_layer == "L23" and to_layer == "L23":
                                ## L2/3層同士のシナプス接続を作成
                                for from_l23_neuron_type in l23_neuron_types:
                                    for to_l23_neuron_type in l23_neuron_types:
                                        w_ave, w_std = self.load_synaptic_weight(from_layer, from_l23_neuron_type, to_l23_neuron_type, from_exc_or_inh)
                                        delay_ave, delay_std = self.load_synaptic_delay(from_exc_or_inh)
                                        decay_time = self.load_synaptic_decay_time(from_l23_neuron_type, to_l23_neuron_type, from_exc_or_inh, to_exc_or_inh)
                                        ### 接続確率を取得
                                        if from_l23_neuron_type == "pyr" and to_l23_neuron_type == "sst": # pyr -> sst
                                            p = params["synapse"]["conn_probability"]["3inh"]["pyr->sst"]
                                        elif from_l23_neuron_type == "pyr" and to_l23_neuron_type == "pv": # pyr -> pv
                                            p = params["synapse"]["conn_probability"]["3inh"]["pyr->pv"]
                                        elif from_l23_neuron_type == "pv" and to_l23_neuron_type == "pyr": # pv -> pyr
                                            p = params["synapse"]["conn_probability"]["3inh"]["pv->pyr"]
                                        elif from_l23_neuron_type == "pyr" and to_l23_neuron_type == "pyr": # pyr -> pyr
                                            p = params["synapse"]["conn_probability"]["3inh"]["pyr->pyr"]
                                        else:
                                            p = params["synapse"]["conn_probability"][f"{from_exc_or_inh}->{to_exc_or_inh}"][from_layer][to_layer]
                                        try:
                                            weighting_factor = params["synapse"]["weighting_factor"][f"{from_l23_neuron_type}->{to_l23_neuron_type}"]
                                        except KeyError as e:
                                            weighting_factor = 1
                                        obj[f"{from_layer}_S_{from_l23_neuron_type}_to_{to_layer}_{to_l23_neuron_type}"] = Normal_Synapse(obj[f"{from_layer}_N_{from_l23_neuron_type}"], 
                                                                                                                        obj[f"{to_layer}_N_{to_l23_neuron_type}"], 
                                                                                                                        name=f"M{circuit_id}_{from_layer}_S_{from_l23_neuron_type}_to_{circuit_id}_{to_layer}_{to_l23_neuron_type}", 
                                                                                                                        connect=True, 
                                                                                                                        p=p, # 接続確率
                                                                                                                        params={"w_ave": w_ave,
                                                                                                                                "w_std": w_std,
                                                                                                                                "weighting_factor": weighting_factor,
                                                                                                                                "delay_ave": delay_ave,
                                                                                                                                "delay_std": delay_std,
                                                                                                                                "tau": decay_time},
                                                                                                                        exc_or_inh=from_exc_or_inh) 
                            elif from_layer == "L23" and not to_layer == "L23" :
                                ## L2/3層からのシナプス接続を作成
                                for from_inh_neuron_type in l23_neuron_types:
                                    w_ave, w_std = self.load_synaptic_weight(from_layer, from_inh_neuron_type, to_neuron_type, from_exc_or_inh)
                                    delay_ave, delay_std = self.load_synaptic_delay(from_exc_or_inh)
                                    decay_time = self.load_synaptic_decay_time(from_inh_neuron_type, to_neuron_type, from_exc_or_inh, to_exc_or_inh)
                                    obj[f"{from_layer}_S_{from_inh_neuron_type}_to_{to_layer}_{to_neuron_type}"] = Normal_Synapse(obj[f"{from_layer}_N_{from_inh_neuron_type}"], 
                                                                                                                        obj[f"{to_layer}_N_{to_neuron_type}"], 
                                                                                                                        name=f"M{circuit_id}_{from_layer}_S_{from_inh_neuron_type}_to_{circuit_id}_{to_layer}_{to_neuron_type}", 
                                                                                                                        connect=True, 
                                                                                                                        p=params["synapse"]["conn_probability"][f"{from_exc_or_inh}->{to_exc_or_inh}"][from_layer][to_layer], # 接続確率
                                                                                                                        params={"w_ave": w_ave,
                                                                                                                                "w_std": w_std,
                                                                                                                                "delay_ave": delay_ave,
                                                                                                                                "delay_std": delay_std,
                                                                                                                                "tau": decay_time},
                                                                                                                        exc_or_inh=from_exc_or_inh)
                            elif to_layer == "L23" and not from_layer == "L23" :
                                ## L2/3層へのシナプス接続を作成
                                for to_l23_neuron_type in l23_neuron_types:
                                    w_ave, w_std = self.load_synaptic_weight(from_layer, from_neuron_type, to_l23_neuron_type, from_exc_or_inh)
                                    delay_ave, delay_std = self.load_synaptic_delay(from_exc_or_inh)
                                    decay_time = self.load_synaptic_decay_time(from_neuron_type, to_l23_neuron_type, from_exc_or_inh, to_exc_or_inh)
                                    obj[f"{from_layer}_S_{from_neuron_type}_to_{to_layer}_{to_l23_neuron_type}"] = Normal_Synapse(obj[f"{from_layer}_N_{from_neuron_type}"], 
                                                                                                                        obj[f"{to_layer}_N_{to_l23_neuron_type}"], 
                                                                                                                        name=f"M{circuit_id}_{from_layer}_S_{from_neuron_type}_to_{circuit_id}_{to_layer}_{to_l23_neuron_type}", 
                                                                                                                        connect=True, 
                                                                                                                        p=params["synapse"]["conn_probability"][f"{from_exc_or_inh}->{to_exc_or_inh}"][from_layer][to_layer], # 接続確率
                                                                                                                        params={"w_ave": w_ave,
                                                                                                                                "w_std": w_std,
                                                                                                                                "delay_ave": delay_ave,
                                                                                                                                "delay_std": delay_std,
                                                                                                                                "tau": decay_time},
                                                                                                                        exc_or_inh=from_exc_or_inh)
                            else:
                                # その他のシナプス接続を作成
                                w_ave, w_std = self.load_synaptic_weight(from_layer, from_neuron_type, to_neuron_type, from_exc_or_inh)
                                delay_ave, delay_std = self.load_synaptic_delay(from_exc_or_inh)
                                decay_time = self.load_synaptic_decay_time(from_neuron_type, to_neuron_type, from_exc_or_inh, to_exc_or_inh)
                                obj[f"{from_layer}_S_{from_neuron_type}_to_{to_layer}_{to_neuron_type}"] = Normal_Synapse(obj[f"{from_layer}_N_{from_neuron_type}"], 
                                                                                                                        obj[f"{to_layer}_N_{to_neuron_type}"], 
                                                                                                                        name=f"M{circuit_id}_{from_layer}_S_{from_neuron_type}_to_{circuit_id}_{to_layer}_{to_neuron_type}", 
                                                                                                                        connect=True,
                                                                                                                        p=params["synapse"]["conn_probability"][f"{from_exc_or_inh}->{to_exc_or_inh}"][from_layer][to_layer], # 接続確率
                                                                                                                        params={"w_ave": w_ave,
                                                                                                                                "w_std": w_std,
                                                                                                                                "delay_ave": delay_ave,
                                                                                                                                "delay_std": delay_std,
                                                                                                                                "tau": decay_time},
                                                                                                                        exc_or_inh=from_exc_or_inh)
                        except KeyError as e:
                            print(f"シナプス接続が存在しません: {from_layer}_S_{from_neuron_type}_to_{to_layer}_{to_neuron_type}")
                            print(e)
                            continue
        # モニター作成
        for key in params["monitor"].keys():
            try:
                self.network.add(SpikeMonitor(obj[key], record=params["monitor"][key], name=f"M{circuit_id}_spikemon_{key}"))
                if not isinstance(obj[key], Poisson_Input_Neuron):
                    self.network.add(StateMonitor(obj[key], ["Ie", "Ii", "v"], record=0, name=f"M{circuit_id}_statemon_{key}"))
            except KeyError:
                print(f"モニターが存在しません: {key}")
                continue
        for key in params["population_monitor"].keys():
            try:
                self.network.add(PopulationRateMonitor(obj[key], name=f"M{circuit_id}_popmon_{key}"))
            except KeyError:
                print(f"ポピュレーションモニターが存在しません: {key}")
                continue
        self.network.add(obj.values())
        
    def load_synaptic_weight(self, from_layer:str=None, from_neuron_type:str=None, to_neuron_type:str=None, from_exc_or_inh:str=None):
        if from_layer == "ex_fibers":
            w_ave = self.params["ex_fibers"]["weight"]["ave"]
            w_std = self.params["ex_fibers"]["weight"]["std"]
        elif from_layer == "L4" and from_neuron_type == "exc" and to_neuron_type == "pyr":
            w_ave = self.params["synapse"]["weight"]["l4e->pyr"]["ave"]
            w_std = self.params["synapse"]["weight"]["l4e->pyr"]["std"]
        else:
            try:
                w_ave = self.params["synapse"]["weight"][f"{from_neuron_type}->{to_neuron_type}"]["ave"]
                w_std = self.params["synapse"]["weight"][f"{from_neuron_type}->{to_neuron_type}"]["std"]
            except KeyError:
                # print(f"デフォルトの重みが使用されます: {from_neuron_type}->{to_neuron_type}")
                if from_exc_or_inh == "exc":
                    w_ave = self.params["synapse"]["weight"]["default_exc"]["ave"]
                    w_std = self.params["synapse"]["weight"]["default_exc"]["std"]  
                elif from_exc_or_inh == "inh":
                    w_ave = self.params["synapse"]["weight"]["default_inh"]["ave"]
                    w_std = self.params["synapse"]["weight"]["default_inh"]["std"]
                else:
                    raise ValueError("from_neuron_typeが不正です。")
        return w_ave, w_std
    
    def load_synaptic_decay_time(self, from_neuron_type:str, to_neuron_type:str, from_exc_or_inh:str, to_exc_or_inh:str):
        try:
            decay_time = self.params["synapse"]["decay_time"][f"{from_neuron_type}->{to_neuron_type}"]
        except KeyError:
            # print(f"シナプス減衰時間がデフォルトに設定されます: {from_neuron_type}->{to_neuron_type}")
            if from_exc_or_inh == "exc": 
                decay_time = self.params["synapse"]["decay_time"]["default_exc"]
            elif from_exc_or_inh == "inh":
                decay_time = self.params["synapse"]["decay_time"]["default_inh"]
            else:
                raise ValueError("from_exc_or_inhが不正です。")
        assert decay_time is not None, "シナプス減衰時間が設定されていません。"
        return decay_time
    

    def load_synaptic_delay(self, from_exc_or_inh:str=None):
        if from_exc_or_inh == "exc":
            delay_ave = self.params["synapse"]["delay"]["intra_column_exc"]["ave"]
            delay_std = self.params["synapse"]["delay"]["intra_column_exc"]["std"]
        elif from_exc_or_inh == "inh":
            delay_ave = self.params["synapse"]["delay"]["intra_column_inh"]["ave"]
            delay_std = self.params["synapse"]["delay"]["intra_column_inh"]["std"]
        else:
            raise ValueError("from_exc_or_inhが不正です。")
        assert delay_ave is not None, "シナプス遅延が設定されていません。"
        return delay_ave, delay_std
    
    def set_input_neuron_rate(self, rate:float):
        """
        入力ニューロンの発火率を設定する関数
        
        Args:
            rate (float): 発火率（Hz）
        """
        # rateに単位がなかったら単位をつける
        if isinstance(rate, (int, float)):
            rate = rate * Hz
        self.network[f"M{self.circuit_id}_N_inp"].set_rate(rate)

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
        self.obj["S_1"] = Normal_Synapse(self.obj["N_1"], self.obj["N_2"], exc_or_inh="exc", name="S_1", connect="i==j", params=params["static_synapse_params_ei"]) # 興奮ニューロンから抑制ニューロン
        self.obj["S_2"] = Normal_Synapse(self.obj["N_2"], self.obj["N_1"], exc_or_inh="inh", name="S_2", connect="i!=j", params=params["static_synapse_params_ie"]) # 側抑制
        
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
        params = load_parameters(params_json_path)
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
        params = load_parameters(params_json_path)
        
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
            

        





