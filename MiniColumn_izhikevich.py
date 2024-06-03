from brian2 import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

defaultclock.dt = 1 * ms


class MiniColumn:
    """
    ミニカラムモデル．
    
    n_l4(int): L4層のニューロン数
    
    n_l23(int): L2/3層のニューロン数
    
    time_profile(TimedArray): 時間プロファイル
    
    synapse_between_same_layer(bool): 同じ層間のシナプスを作成するかどうか
    """
    def __init__(self, n_l4, n_l23, time_profile: TimedArray = None, synapse_between_same_layer=False, neuron_model="LIF"):
        # Common Parameters
        v_rest = -65  # Resting potential
        v_th = -40  # Threshold voltage
        
        if neuron_model == "LIF": # Leaky Integrate-and-Fire
            eqs_neuron_model = """
            dv/dt = ((v_rest - v) + I) / tau : 1
            v_rest : 1
            I : 1
            tau : second
            """
            neuron_params = {
                "threshold": -40, 
                "reset": -65,
                "refractory": "2 * ms",
                "method": "euler",
                "tau": 10,
                "I": 50,
                } # Neuron parameters
                
            
        elif neuron_model == "Izhikevich": # Izhikevich
            ## Izhikevich
            tau = 50 * ms # Time constant of the current decay
            # ==== Izhikevich2003モデル ====    
            # a : uのスケーリング係数
            # b : vに対してuをどれくらい変化させるか
            # c : vの静止膜電位
            # d : 発火後に静止膜電位に戻るまでの時間を変化させる定数
            eqs_neuron_model = """    
            dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1 (unless refractory)
            du/dt = (a*(b*v - u))/ms : 1
            dI/dt = -I / tau : 1
            I_noise : 1
            a : 1
            b : 1
            c : 1
            d : 1
            tau : second
            """
            izhi_params = {"RS": {"a": 0.02, "b": 0.2, "c": -65, "d": 8}, # Regular spiking
                        "IB": {"a": 0.02, "b": 0.2, "c": -55, "d": 4}, # Intermittent burst
                        "CH": {"a": 0.02, "b": 0.2, "c": -50, "d": 2}, # Chattering
                        "FS": {"a": 0.1, "b": 0.2, "c": -65, "d": 2}, # Fast spiking
                        "TC": {"a": 0.02, "b": 0.25, "c": -65, "d": 0.05}, # Transient change
                        "LTS": {"a": 0.02, "b": 0.25, "c": -65, "d": 2},} # Long-term synaptic plasticity
            
            select_neuron_type = "RS" # Select the type of neuron

            
        ## STDP
        wmax = 1 # Maximum value of the synaptic weight
        w = np.ones((n_l4, n_l23)) # Synaptic weight
        Apre = 0.01 # Maximum increase of the synapse trace of the pre-neuron
        Apost = -0.01 # Maximum decrease of the synapse trace of the post-neuron
        taupost = 20 * ms # Time constant of the spike trace of the post-neuron
        taupre = 20 * ms # Time constant of the spike trace of the pre-neuron
              
        # Synapse equations
        ## STDP model
        eqs_stdp = """
        w : 1
        Apre : 1
        Apost : 1
        taupost : second
        taupre : second
        wmax : 1
        dapre/dt = -apre/taupre : 1 (clock-driven)
        dapost/dt = -apost/taupost : 1 (clock-driven)
        """
        eqs_stdp_on_pre = """
        I_post += w
        apre += Apre
        w = clip(w + apost, 0, wmax)
        """
        eqs_stdp_on_post = """
        apost += Apost
        w = clip(w + apre, 0, wmax)
        """
        # In brian2, 
        # v_post : voltage of the post-synaptic neuron
        # I_post : current of the post-synaptic neuron
        
        
        self.N = {}
        self.S = {}  
        self.spikemon = {} 
        self.statemon = {}  
               
        # Neuron group of L4
        self.N["l4"] = NeuronGroup(
            n_l4,
            eqs_neuron_model,
            threshold="v>=-30",
            reset="v=-65",
            refractory="2 * ms",
            method=neuron_params["method"],
        )

        # Neuron group of L2/3
        self.N["l23"] = NeuronGroup(
            n_l23,
            eqs_neuron_model,
            threshold="v>=-30",
            reset="v=-65",
            refractory="2 * ms",
            method=neuron_params["method"],
        )

        # Neuron settings
        if neuron_model == "Izhikevich":
            self.N["l4"].a = izhi_params[select_neuron_type]["a"]
            self.N["l4"].b = izhi_params[select_neuron_type]["b"]
            self.N["l4"].c = izhi_params[select_neuron_type]["c"]
            self.N["l4"].d = izhi_params[select_neuron_type]["d"]
            self.N["l4"].tau = tau  # 電流減衰の時定数

            self.N["l23"].a = izhi_params[select_neuron_type]["a"]
            self.N["l23"].b = izhi_params[select_neuron_type]["b"]
            self.N["l23"].c = izhi_params[select_neuron_type]["c"]
            self.N["l23"].d = izhi_params[select_neuron_type]["d"]
            self.N["l23"].tau = tau  # 電流減衰の時定数
        elif neuron_model == "LIF":
            self.N["l4"].v_rest = neuron_params["reset"]
            self.N["l4"].I = neuron_params["I"]
            self.N["l4"].tau = neuron_params["tau"]

        # Synapse settings
        ## L4 -> L2/3
        self.S["l4_23"] = Synapses(
            self.N["l4"],
            self.N["l23"],
            model=eqs_stdp,
            on_pre=eqs_stdp_on_pre,
            on_post=eqs_stdp_on_post,
            delay=1 * ms,
        )
        ## L4 -> L4
        if synapse_between_same_layer:
            self.S["l4_4"] = Synapses(
                self.N["l4"],
                self.N["l4"],
                model=eqs_stdp,
                on_pre=eqs_stdp_on_pre,
                on_post=eqs_stdp_on_post,
                delay=1 * ms,
            )
            self.S["l4_4"].connect(condition="i!=j")
            
        self.S["l4_23"].connect()
        
        for S, w in zip(self.S, w):
            self.S[S].w = w
            self.S[S].Apre = Apre
            self.S[S].Apost = Apost
            self.S[S].taupre = taupre
            self.S[S].taupost = taupost
            self.S[S].wmax = wmax

        # Time profileで刺激を与える場合
        if time_profile is not None:
            self.N["l4"].run_regularly("I = time_profile(t)")        
        
        # Monitor settings
        self.spikemon["l4"] = SpikeMonitor(self.N["l4"])
        self.spikemon["l23"] = SpikeMonitor(self.N["l23"])
        self.statemon["l4"] = StateMonitor(
            self.N["l4"], ["v", "I"], record=True, when="after_thresholds"
        )
        self.statemon["l23"] = StateMonitor(
            self.N["l23"], ["v", "I"], record=True, when="after_thresholds"
        )
        self.statemon["S_l4_23"] = StateMonitor(
            self.S["l4_23"], ["w", "apre", "apost"], record=True, when="after_thresholds"
        )

        self.network = Network(
            self.N, self.S, self.spikemon, self.statemon
        )
        
        

    def run(self, duration):
        self.network.run(duration)

    def draw_plot_and_graph(self, title=None):
        fig, (
            ax_l4_spike,
            ax_l4_apre,
            ax_l4_apost,
            ax_l4_voltage,
            ax_l4_current,
            ax_l23_spike,
            ax_l23_voltage,
            ax_l23_current,
        ) = plt.subplots(8, 1, sharex=True, figsize=(12, 9))
        plt.subplots_adjust(hspace=1.0)

        # スパイクのプロット
        ax_l4_spike.scatter(
            self.spikemon["l4"].t / ms, self.spikemon["l4"].i, s=2, color="k", label="L4"
        )
        ax_l4_spike.set_ylabel("Neuron number")
        ax_l4_spike.set_title("L4")

        ax_l23_spike.scatter(
            self.spikemon["l23"].t / ms, self.spikemon["l23"].i, s=2, color="k", label="L2/3"
        )
        ax_l23_spike.set_ylabel("Neuron number")
        ax_l23_spike.set_title("L2/3")
        
        # スパイクトレースのプロット
        ax_l4_apre.plot(self.statemon["S_l4_23"].t / ms, self.statemon["S_l4_23"].apre[0], color="k")
        ax_l4_apost.plot(self.statemon["S_l4_23"].t / ms, self.statemon["S_l4_23"].apost[0], color="k")
        ax_l4_apre.set_ylabel("Apre")
        ax_l4_apost.set_ylabel("Apost")

        # 膜電位のプロット
        ax_l4_voltage.plot(
            self.statemon["l4"].t / ms,
            self.statemon["l4"].v[0],
            color="k",
        )
        ax_l4_voltage.set_ylabel("Voltage (mV)")
        ax_l4_voltage.set_ylim(-100, 30)
        ax_l4_voltage.axhline(
            -65, color="red", linewidth=0.5, linestyle="--", label="Resting Potential"
        )
        ax_l4_voltage.legend()

        ax_l23_voltage.plot(
            self.statemon["l23"].t / ms,
            self.statemon["l23"].v[0],
            color="k",
        )
        ax_l23_voltage.set_ylabel("Voltage (mV)")
        ax_l23_voltage.set_ylim(-100, 30)
        ax_l23_voltage.set_xlabel("Time (ms)")
        ax_l23_voltage.axhline(
            -65, color="red", linewidth=0.5, linestyle="--", label="Resting Potential"
        )
        ax_l23_voltage.legend()

        # 電流のプロット
        ax_l4_current.plot(self.statemon["l4"].t / ms, self.statemon["l4"].I[0], color="k")
        ax_l4_current.set_ylabel("Current (pA)")
        ax_l4_current.set_xlabel("Time (ms)")
        ax_l4_current.set_ylim(0, 30)

        ax_l23_current.plot(self.statemon["l23"].t / ms, self.statemon["l23"].I[0], color="k")
        ax_l23_current.set_ylabel("Current (pA)")
        ax_l23_current.set_xlabel("Time (ms)")
        ax_l23_current.set_ylim(0, 30)
        
        # シナプス重みのプロット
        fig2, ax_stdp_w = plt.subplots(5, 1, figsize=(12, 10))
        for i in range(len(ax_stdp_w)):
            ax_stdp_w[i].plot(self.statemon["S_l4_23"].t / ms, self.statemon["S_l4_23"].w[i], color="k")
            ax_stdp_w[i].set_ylabel(f"Weight[{i}]")
            ax_stdp_w[i].set_xlabel("Time (ms)")
            
        if title is not None:
            fig.suptitle(title + " Current, Voltage, and Spikes Over Time of Neuron No.[0]")
            fig2.suptitle(title + " Synapse Weights Over Time")
        else:
            fig.suptitle("Neural Activity and Voltage Plots")
            fig2.suptitle("Synapse Weights Over Time")
        
        # 発火率
        print("Fire rate(%s):" % title)
        print("L4: ", self.spikemon["l4"].num_spikes / len(self.N["l4"]) / second)
        print("L2/3: ", self.spikemon["l23"].num_spikes / len(self.N["l23"]) / second)
        
        return fig
        
class ScrollableWindow(QMainWindow):
    def __init__(self, fig):
        super().__init__()
        # ウィジェットの設定
        self.scroll = QScrollArea(self)  # スクロールエリアを作成
        self.widget = QWidget()          # スクロールエリアに入れるウィジェット
        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.scroll.setWidget(self.canvas)
        self.setCentralWidget(self.scroll)
        
        self.resize(1200, 850)
        self.show()

if __name__ == "__main__":
    pass
