from brian2 import Synapses
from brian2.units import *
import pprint as pp


class STDP:
    """
    STDPシナプスを作成します。
    """

    def __init__(self, params=None):

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
            raise ValueError("シナプス作成時にパラメータを指定してください。")
        else:
            self.params = params

    def __call__(
        self, pre_neurons, post_neurons, name:str, connect=True
    ):

        
        synapse = Synapses(
            source=pre_neurons,
            target=post_neurons,
            model=self.model,
            on_pre=self.on_pre,
            on_post=self.on_post,
            name=name,
            namespace=self.params,
            method="euler"
        )
        synapse.connect(connect)
        synapse.w = "rand() * (wmax - wmin) + wmin"
        return synapse


class NonSTDP:
    """
    非STDPシナプスを作成します。
    """

    def __init__(self, params=None):

        self.model = "w : 1"

        self.on_pre_e = "ge_post += w" # 後ニューロンへの興奮性入力
        self.on_pre_i = "gi_post += w" # 後ニューロンへの抑制性入力

        if params is None:
            # パラメータ未指定時のデフォルトのパラメータ
            self.params = {
                "w": 1,  # 固定重み
            }
        else:
            self.params = params

    def __call__(self, pre_neurons, post_neurons, exc_or_inh:str, name:str, delay=1*ms, connect=True):
        if exc_or_inh == "exc":
            on_pre = self.on_pre_e
        elif exc_or_inh == "inh":
            on_pre = self.on_pre_i
        else:
            raise ValueError("exc_or_inh must be 'exc' or 'inh'")
        
        synapse = Synapses(
            source=pre_neurons,
            target=post_neurons,
            model=self.model,
            on_pre=on_pre,
            method="euler",
            name=name,
            delay=delay
        )
        synapse.connect(connect)
        synapse.w = self.params["w"]
        return synapse
