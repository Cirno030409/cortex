import pprint

import Network.Neurons as Neurons
import Network.Synapses as MySynapses
from brian2 import *


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