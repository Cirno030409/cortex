import Network.Neurons as Neurons
import Network.Synapses as MySynapses
from brian2 import *


class MiniColumn:
    """
    ミニカラムネットワークを作成します。
    """
    def __init__(self, n_inp:int, n_l4:int, n_l23:int, n_inh:int, monitors:bool):
        obj = {}
        self.n_l4 = n_l4
        self.n_l23 = n_l23
        self.n_inp = n_inp
        
        neuron = Neurons.Conductance_Izhikevich2003(neuron_type="RS")
        stdp_synapse = MySynapses.STDP()
        nonstdp_synapse = MySynapses.NonSTDP()
        
        obj["N_input"] = PoissonGroup(n_inp, rates=np.zeros(n_inp) * Hz, name="N_input")
        
        obj["N_l4"] = neuron(N=n_l4, exc_or_inh="exc", tag_name="N_l4")
        
        obj["N_l23"] = neuron(N=n_l23, exc_or_inh="exc", tag_name="N_l23")
        
        # obj["inhibitory"] = neuron(N=n_inh, exc_or_inh="inh", tag_name="N_inh")
        
        obj["S_input_l4"] = stdp_synapse(obj["N_input"], obj["N_l4"], connect="i==j", tag_name="S_input_l4")
        
        obj["S_l4_l23"] = stdp_synapse(obj["N_l4"], obj["N_l23"], connect=True, tag_name="S_l4_l23")
        
        if monitors:
            for i in list(obj.keys()):
                if i == "S_input_l4" or i == "S_l4_l23":
                    obj["statemon_"+i] = StateMonitor(obj[i], ["w", "apre", "apost", "gsyn"], record=True, name="statemon_"+i)
                elif i == "N_l4" or i == "N_l23":
                    obj["spikemon_"+i] = SpikeMonitor(obj[i], record=True, name="spikemon_"+i)
                    obj["statemon_"+i] = StateMonitor(obj[i], ["v", "I"], record=True, name="statemon_"+i)
                    obj["popmon_"+i] = PopulationRateMonitor(obj[i], name="popmon_"+i)
                elif i == "N_input":
                    obj["spikemon_"+i] = SpikeMonitor(obj[i], record=True, name="spikemon_"+i)
            
        self.net = Network(obj.values())
        self.net.run(0 * second)
        
    def __getitem__(self, key):
        return self.net[key]
    
    def __setitem__(self, key, value):
        self.net[key] = value
        
    def run(self, inp:np.ndarray, duration:float):
        """
        ネットワークを実行します。

        Args:
            inp (np.ndarray): ネットワークに入力する発火レートを指定します。
            duration (float): ネットワークの実行時間を指定します。
        """
        self.net["N_input"].rates = inp
        self.net.run(duration)
        
    def reset(self):
        self.net.reset()
        self.net.run(0 * second)