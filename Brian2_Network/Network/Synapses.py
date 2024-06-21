from brian2 import Synapses
from brian2.units import *


class STDP:
    """
    STDPシナプスを作成します。
    """

    def __init__(self):

        self.model = """
            dapre/dt = (-apre - alpha)/taupre : 1 (clock-driven)
            dapost/dt = (-apost)/taupost : 1 (clock-driven)
            w : 1
        """

        self.on_pre = """
            gsyn_post += w
            apre = Apre
            w = clip(w + apost, 0, wmax)
        """

        self.on_post = """
            apost = Apost
            w = clip(w + apre, 0, wmax)
        """

        self.params = {
            "wmax": 1,  # 最大重み
            "wmin": 0,  # 最小重み
            "Apre": 0.01,  # 前ニューロンのスパイクトレースのリセット値
            "Apost": 1,  # 後ニューロンのスパイクトレースのリセット値
            "taupre": 20 * ms,  # 前ニューロンのスパイクトレースの時定数
            "taupost": 20 * ms,  # 後ニューロンのスパイクトレースの時定数
            "alpha": 0.01,  # スパイクトレースの収束地点
        }

    def __call__(
        self, pre_neurons, post_neurons, tag_name: str, connect=True
    ):
        synapse = Synapses(
            source=pre_neurons,
            target=post_neurons,
            model=self.model,
            on_pre=self.on_pre,
            on_post=self.on_post,
            name=tag_name,
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

    def __init__(self):

        self.model = "w : 1"

        self.on_pre = "gsyn_post += w"

        self.params = {
            "w": 1.0,
        }

    def __call__(self, pre_neurons, post_neurons, tag_name: str, connect=True):
        synapse = Synapses(
            source=pre_neurons,
            target=post_neurons,
            model=self.model,
            on_pre=self.on_pre,
            method="euler",
            name=tag_name,
            namespace=self.params,
        )
        synapse.connect(connect)
        if self.w is not None:
            synapse.w = self.w
        else:
            synapse.w = 1
        return synapse
