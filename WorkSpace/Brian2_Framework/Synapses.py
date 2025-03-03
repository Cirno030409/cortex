from brian2 import Synapses
from brian2 import NeuronGroup
from brian2.units import *
import pprint as pp


class STDP_Synapse(Synapses):
    """
    STDPシナプスを作成します。
    """

    def __init__(self, pre_neurons:NeuronGroup, post_neurons:NeuronGroup, name:str, connect=True, params=None, exc_or_inh:str="exc", p=1.0, *args, **kwargs):
        """
        pre_neurons: 前ニューロン
        post_neurons: 後ニューロン
        name: シナプスの名前
        connect: シナプスの接続方法
        params: パラメータの辞書
        *args, **kwargs: その他の引数
        """
        self.exc_or_inh = exc_or_inh
        self.p = p
        self.model = """
            dapre/dt = (-apre - alpha)/taupre : siemens (event-driven)
            dapost/dt = (-apost)/taupost : siemens (event-driven)

            w : siemens
        """
        
        if exc_or_inh == "exc":
            self.on_pre = """
                apre = Apre
                w = clip(w - apost * nu_post * sw, wmin, wmax)
                ge_post += w * ge_gain
            """
        elif exc_or_inh == "inh":
            self.on_pre = """
                apre = Apre
                w = clip(w - apost * nu_post * sw, wmin, wmax)
                gi_post += w * gi_gain
            """
        else:
            raise ValueError("STDPシナプスを作成するときは，'exc'か'inh'を指定してください。")

        self.on_post = """
            apost = Apost
            w = clip(w + apre * nu_pre * sw, wmin, wmax)
        """
        if params is None:
            raise ValueError("シナプス作成時にパラメータの辞書を渡してください。")
        else:
            self.params = params
        
        super().__init__(pre_neurons, post_neurons, model=self.model, on_pre=self.on_pre, on_post=self.on_post, namespace=dict(self.params), name=name, *args, **kwargs)
        self.connect(connect, p=p)
        self.w = "rand() * (wmax - wmin) + wmin"

class Normal_Synapse(Synapses):

    """
    非STDPシナプスを作成します。
    """

    def __init__(self, pre_neurons:NeuronGroup, post_neurons:NeuronGroup, name:str, connect=True, params=None, exc_or_inh:str="", p=1.0, *args, **kwargs):
        """
        学習を行わない非STDPシナプスを作成します。
        
        Args:
        pre_neurons: 前ニューロン
        post_neurons: 後ニューロン
        exc_or_inh: 興奮性シナプスか抑制性シナプスか
        name: シナプスの名前
        connect: シナプスの接続方法
        params: パラメータの辞書
        *args, **kwargs: その他の引数
        """
        if params is None:
            raise ValueError("シナプス作成時にパラメータの辞書を渡してください。")
        else:
            self.params = dict(params) # 参照関係を切る
        self.exc_or_inh = exc_or_inh
        self.p = p

        # post_neuronがConductance BasedかCurrent Basedかで挙動を変える
        is_conductance = hasattr(post_neurons, "ge")
        if is_conductance: # Conductance Based
            self.model = """
                w : siemens
            """
            if exc_or_inh == "exc":
                self.on_pre = "ge_post += w" # 後ニューロンへの興奮性入力
            elif exc_or_inh == "inh":
                self.on_pre = "gi_post += w" # 後ニューロンへの抑制性入力
            else:
                raise ValueError("通常のシナプスを作成するときは，'exc'か'inh'を指定してください。")
        else: # Current Based
            if exc_or_inh == "exc": # 興奮性シナプス
                self.model = """
                    dsyn/dt = -syn/tau : amp (clock-driven)
                    w : amp
                """ # (summed)は後ニューロンの変数に合計される
                self.on_pre = "syn = w" # シナプストレースを固定値に増加
            elif exc_or_inh == "inh": # 抑制性シナプス
                self.model = """
                    dsyn/dt = -syn/tau : amp (clock-driven)
                    w : amp
                """
                self.on_pre = "syn = w" # シナプストレースを固定値に増加
            else:
                raise ValueError("通常のシナプスを作成するときは，'exc'か'inh'を指定してください。")
        
        if "w_ave" in params.keys() or "w_std" in params.keys() :
            self.w_ave = params["w_ave"]
            self.w_std = params["w_std"]
            if "w_ave" in params.keys():
                del params["w_ave"]
            if "w_std" in params.keys():
                del params["w_std"]
        else:
            self.w_ave = params["w"]
            if is_conductance:
                self.w_std = 0*nS
            else:
                self.w_std = 0*pA
        
        if "delay_ave" in params.keys() or "delay_std" in params.keys() :
            self.delay_ave = params["delay_ave"]
            self.delay_std = params["delay_std"]
            if "delay_ave" in params.keys():
                del params["delay_ave"]
            if "delay_std" in params.keys():
                del params["delay_std"]
        else:
            self.delay_ave = params["delay"]
            self.delay_std = 0*ms
        
        super().__init__(pre_neurons, post_neurons, model=self.model, on_pre=self.on_pre, method="euler", namespace=dict(self.params), name=name, *args, **kwargs)
        self.connect(connect, p=p)
        if is_conductance:
            self.w = f"({self.w_ave/nS} + {self.w_std/nS} * randn()) * siemens"
        else:
            self.w = f"({self.w_ave/pA} + {self.w_std/pA} * randn()) * pA"
        self.delay = f"({self.delay_ave/ms} + {self.delay_std/ms} * randn()) * ms"

        # 毎ステップでニューロンにシナプストレースを加算する
        if not is_conductance:  # Current Based の場合
            # NOTE order=0でモニターが値を取得するので，計算はorder=0には終わらせておく
            self.run_regularly("Ie_post = 0*pA", order=-2)
            self.run_regularly("Ii_post = 0*pA", order=-2)
            if exc_or_inh == "exc":
                self.run_regularly("Ie_post += syn", order=-1)
            elif exc_or_inh == "inh":
                self.run_regularly("Ii_post += syn", order=-1)
