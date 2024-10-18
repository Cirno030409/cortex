from brian2 import Synapses
from brian2 import NeuronGroup
from brian2.units import *
import pprint as pp


class STDP_Synapse(Synapses):
    """
    STDPシナプスを作成します。
    """

    def __init__(self, pre_neurons:NeuronGroup, post_neurons:NeuronGroup, name:str, connect=True, params=None, *args, **kwargs):
        """
        pre_neurons: 前ニューロン
        post_neurons: 後ニューロン
        name: シナプスの名前
        connect: シナプスの接続方法
        params: パラメータの辞書
        *args, **kwargs: その他の引数
        """

        self.model = """
            dapre/dt = (-apre - alpha)/taupre : 1 (event-driven)
            dapost/dt = (-apost)/taupost : 1 (event-driven)
            w : 1
        """

        self.on_pre = """
            apre = Apre
            w = clip(w - apost * nu_post * sw, 0, wmax)
            ge_post += w * g_gain
        """

        self.on_post = """
            apost = Apost
            w = clip(w + apre * nu_pre * sw, 0, wmax)
        """
        if params is None:
            raise ValueError("シナプス作成時にパラメータの辞書を渡してください。")
        else:
            self.params = params
        
        super().__init__(pre_neurons, post_neurons, model=self.model, on_pre=self.on_pre, on_post=self.on_post, namespace=self.params, name=name, *args, **kwargs)
        self.connect(connect)
        self.w = "rand() * (wmax - wmin) + wmin"

class Normal_Synapse(Synapses):

    """
    非STDPシナプスを作成します。
    """

    def __init__(self, pre_neurons:NeuronGroup, post_neurons:NeuronGroup, exc_or_inh:str, name:str, connect=True, params=None, *args, **kwargs):
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
        self.model = "w : 1"

        if params is None:
            raise ValueError("シナプス作成時にパラメータの辞書を渡してください。")
        else:
            self.params = params

        if exc_or_inh == "exc":
            self.on_pre = "ge_post += w" # 後ニューロンへの興奮性入力
        elif exc_or_inh == "inh":
            self.on_pre = "gi_post += w" # 後ニューロンへの抑制性入力
        else:
            raise ValueError("通常のシナプスを作成するときは，'exc'か'inh'を指定してください。")

        
        super().__init__(pre_neurons, post_neurons, model=self.model, on_pre=self.on_pre, method="euler", name=name, *args, **kwargs)
        self.connect(connect)
        self.w = self.params["w"]
